using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

class Program
{
    static readonly HttpClient PostClient = new HttpClient(new SocketsHttpHandler
    {
        ConnectTimeout = TimeSpan.FromMilliseconds(300),
        PooledConnectionLifetime = TimeSpan.FromMinutes(2),
        UseProxy = false,
        UseCookies = false,
    })
    {
        Timeout = TimeSpan.FromSeconds(180)
    };

    static readonly HttpClient CheckClient = new HttpClient(new SocketsHttpHandler
    {
        ConnectTimeout = TimeSpan.FromMilliseconds(500),
        PooledConnectionLifetime = TimeSpan.FromMinutes(2),
        UseProxy = false,
        UseCookies = false,
    })
    {
        Timeout = TimeSpan.FromSeconds(2)
    };

    static int TargetPort = 8572;
    static readonly string LauncherScriptName = OperatingSystem.IsWindows() ? "start.bat" : "start.sh";

    sealed class ServerLocation
    {
        public string ServerRoot { get; set; } = "";
        public string PythonExe { get; set; } = "";
        public string ScriptPath { get; set; } = "";
        public string LauncherScript { get; set; } = "";
    }

    static async Task Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine("[Main] No arguments provided.");
            Environment.Exit(1);
        }

        ResolveTargetPortFast();
        string jsonPayload = BuildJsonPayload(args);

        // --- Fast Path: Optimistic single-trip POST / ---
        var (fastSuccess, shouldEnsureServer) = await TryOptimisticPostAsync(jsonPayload);
        if (fastSuccess)
        {
            return;
        }
        if (!shouldEnsureServer)
        {
            Environment.Exit(1);
        }

        // --- Slow Path: Server is down or still warming up (503) ---
        var location = ResolveServerLocation();
        await EnsureServerRunningAsync(location);
        await CommunicateWithServer(jsonPayload);
    }

    static async Task<(bool Succeeded, bool ShouldEnsureServer)> TryOptimisticPostAsync(string jsonPayload)
    {
        var sw = Stopwatch.StartNew();
        try
        {
            using var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");
            using HttpResponseMessage response = await PostClient.PostAsync($"http://127.0.0.1:{TargetPort}/", content);
            if (response.IsSuccessStatusCode)
            {
                return (true, false);
            }

            if (response.StatusCode == HttpStatusCode.ServiceUnavailable)
            {
                return (false, true);
            }

            string responseBody = response.Content == null ? "[No Body]" : await response.Content.ReadAsStringAsync();
            Console.Error.WriteLine($"[Communicate] Error: Status {(int)response.StatusCode} {response.ReasonPhrase} - {responseBody}");
            return (false, false);
        }
        catch (HttpRequestException ex) when (IsConnectionRefusedOrUnreachable(ex) || sw.ElapsedMilliseconds < 1500)
        {
            return (false, true);
        }
        catch (TaskCanceledException) when (sw.ElapsedMilliseconds < 1500)
        {
            // ConnectTimeout (300ms) fired before TCP connection was established
            return (false, true);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[Communicate] Request failed after {sw.ElapsedMilliseconds}ms: {ex.Message}");
            return (false, false);
        }
    }

    static bool IsConnectionRefusedOrUnreachable(HttpRequestException ex)
    {
        if (ex.InnerException is SocketException sockEx)
        {
            return sockEx.SocketErrorCode == SocketError.ConnectionRefused
                || sockEx.SocketErrorCode == SocketError.TimedOut
                || sockEx.SocketErrorCode == SocketError.NetworkUnreachable
                || sockEx.SocketErrorCode == SocketError.HostUnreachable;
        }
        return false;
    }

    static void ResolveTargetPortFast()
    {
        string? portEnv = Environment.GetEnvironmentVariable("HIFISAMPLER_PORT");
        if (!string.IsNullOrWhiteSpace(portEnv) && int.TryParse(portEnv, out int parsedPort) && parsedPort > 0)
        {
            TargetPort = parsedPort;
            return;
        }

        try
        {
            string userHome = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            string beaconPath = Path.Combine(userHome, ".hifisampler", "server.json");
            if (File.Exists(beaconPath))
            {
                string jsonText = File.ReadAllText(beaconPath, Encoding.UTF8);
                using JsonDocument doc = JsonDocument.Parse(jsonText);
                if (doc.RootElement.TryGetProperty("port", out JsonElement portEl) && portEl.TryGetInt32(out int beaconPort) && beaconPort > 0)
                {
                    TargetPort = beaconPort;
                }
            }
        }
        catch
        {
            // Ignore fast-path beacon read errors; full fallback handles reporting
        }
    }

    static async Task EnsureServerRunningAsync(ServerLocation location)
    {
        bool proceedToCommunication = false;
        bool errorOccurred = false;
        string errorMessage = "";

        if (await IsServerReady(TargetPort))
        {
            return;
        }

        Console.WriteLine("[Main] Server not ready. Acquiring startup lock...");

        using var lockStream = await TryAcquireStartupLockAsync(TargetPort, TimeSpan.FromSeconds(5));
        if (lockStream != null)
        {
            try
            {
                if (await IsServerReady(TargetPort))
                {
                    proceedToCommunication = true;
                }
                else if (!IsPortInUse(TargetPort))
                {
                    Console.WriteLine($"[Main] Port {TargetPort} is free. Launching HifiSampler server...");
                    if (string.IsNullOrEmpty(location.ServerRoot))
                    {
                        errorMessage =
                            "Could not locate HifiSampler server directory. " +
                            "Please run start.bat / start.sh (or uv run hifiserver.py) once in the server folder to register ~/.hifisampler/server.json, " +
                            "or set the HIFISAMPLER_HOME environment variable.";
                        errorOccurred = true;
                    }
                    else
                    {
                        try
                        {
                            LaunchServerProcess(location);
                            bool initialWaitOk = await WaitForServerToStart(TargetPort, 45, "[InitialWait]");
                            if (initialWaitOk)
                            {
                                proceedToCommunication = true;
                            }
                            else
                            {
                                errorMessage = "Server launched (warming up in background), but did not become ready within 45s.";
                                errorOccurred = true;
                            }
                        }
                        catch (Exception ex)
                        {
                            errorMessage = $"Error during server launch execution: {ex.Message}";
                            errorOccurred = true;
                        }
                    }
                }
                else
                {
                    Console.WriteLine($"[Main] Port {TargetPort} is in use; waiting for server initialization...");
                }
            }
            catch (Exception ex)
            {
                errorMessage = $"Unexpected error during startup checks: {ex.Message}";
                errorOccurred = true;
            }
        }
        else
        {
            Console.WriteLine("[Main] Another client instance holds the startup lock; waiting for server...");
        }

        if (errorOccurred)
        {
            Console.Error.WriteLine($"[Main] Exiting due to critical error: {errorMessage}");
            Environment.Exit(1);
        }

        if (!proceedToCommunication)
        {
            bool finalReadinessCheck = await WaitForServerToStart(TargetPort, 45, "[FinalWait]");
            if (!finalReadinessCheck)
            {
                Console.Error.WriteLine("[Main] Error: Server failed readiness check within timeout.");
                Environment.Exit(1);
            }
        }
    }

    static ServerLocation ResolveServerLocation()
    {
        string? portEnv = Environment.GetEnvironmentVariable("HIFISAMPLER_PORT");
        if (!string.IsNullOrWhiteSpace(portEnv) && int.TryParse(portEnv, out int parsedPort) && parsedPort > 0)
        {
            TargetPort = parsedPort;
        }

        // 1. Check HIFISAMPLER_HOME environment variable
        string? envHome = Environment.GetEnvironmentVariable("HIFISAMPLER_HOME");
        if (!string.IsNullOrWhiteSpace(envHome) && TryProbeServerDir(envHome, out var fromEnv))
        {
            return fromEnv;
        }

        // 2. Check symlink target of current executable (if linked via ln -s or mklink)
        string? processPath = Environment.ProcessPath;
        if (!string.IsNullOrEmpty(processPath) && File.Exists(processPath))
        {
            try
            {
                FileSystemInfo? linkTarget = File.ResolveLinkTarget(processPath, returnFinalTarget: true);
                if (linkTarget != null)
                {
                    string? targetDir = Path.GetDirectoryName(linkTarget.FullName);
                    if (!string.IsNullOrEmpty(targetDir) && TryProbeServerDir(targetDir, out var fromLink))
                    {
                        return fromLink;
                    }
                }
            }
            catch
            {
                // Ignore if OS/filesystem does not support ResolveLinkTarget
            }
        }

        // 3. Check executable's own directory
        if (TryProbeServerDir(AppContext.BaseDirectory, out var fromBase))
        {
            return fromBase;
        }

        // 4. Check Global Locator Beacon ~/.hifisampler/server.json
        try
        {
            string userHome = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            string beaconPath = Path.Combine(userHome, ".hifisampler", "server.json");
            if (File.Exists(beaconPath))
            {
                string jsonText = File.ReadAllText(beaconPath, Encoding.UTF8);
                using JsonDocument doc = JsonDocument.Parse(jsonText);
                JsonElement root = doc.RootElement;

                if (root.TryGetProperty("port", out JsonElement portEl) && portEl.TryGetInt32(out int beaconPort) && beaconPort > 0)
                {
                    if (string.IsNullOrWhiteSpace(portEnv))
                    {
                        TargetPort = beaconPort;
                    }
                }

                string? serverRoot = root.TryGetProperty("server_root", out JsonElement srEl) ? srEl.GetString() : null;
                if (!string.IsNullOrWhiteSpace(serverRoot) && TryProbeServerDir(serverRoot, out var fromBeacon))
                {
                    if (root.TryGetProperty("python_exe", out JsonElement pyEl))
                    {
                        string? pyExe = pyEl.GetString();
                        if (!string.IsNullOrWhiteSpace(pyExe) && File.Exists(pyExe))
                        {
                            fromBeacon.PythonExe = pyExe;
                        }
                    }
                    return fromBeacon;
                }
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[Locator] Warning reading ~/.hifisampler/server.json: {ex.Message}");
        }

        return new ServerLocation();
    }

    static bool TryProbeServerDir(string dirPath, out ServerLocation location)
    {
        location = new ServerLocation();
        if (string.IsNullOrWhiteSpace(dirPath) || !Directory.Exists(dirPath))
        {
            return false;
        }

        string fullDir = Path.GetFullPath(dirPath);
        string scriptPath = Path.Combine(fullDir, "hifiserver.py");
        string launcherPath = Path.Combine(fullDir, LauncherScriptName);

        if (!File.Exists(scriptPath) && !File.Exists(launcherPath))
        {
            return false;
        }

        string venvPython = OperatingSystem.IsWindows()
            ? Path.Combine(fullDir, ".venv", "Scripts", "python.exe")
            : Path.Combine(fullDir, ".venv", "bin", "python");

        location.ServerRoot = fullDir;
        location.ScriptPath = File.Exists(scriptPath) ? scriptPath : "";
        location.LauncherScript = File.Exists(launcherPath) ? launcherPath : "";
        location.PythonExe = File.Exists(venvPython) ? venvPython : "";

        if (string.IsNullOrEmpty(location.PythonExe))
        {
            try
            {
                string userHome = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                string beaconPath = Path.Combine(userHome, ".hifisampler", "server.json");
                if (File.Exists(beaconPath))
                {
                    string jsonText = File.ReadAllText(beaconPath, Encoding.UTF8);
                    using JsonDocument doc = JsonDocument.Parse(jsonText);
                    JsonElement root = doc.RootElement;
                    string? beaconRoot = root.TryGetProperty("server_root", out JsonElement srEl) ? srEl.GetString() : null;
                    string? beaconPy = root.TryGetProperty("python_exe", out JsonElement pyEl) ? pyEl.GetString() : null;
                    if (!string.IsNullOrWhiteSpace(beaconRoot) && !string.IsNullOrWhiteSpace(beaconPy) && File.Exists(beaconPy))
                    {
                        StringComparison cmp = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
                        string normBeaconRoot = Path.GetFullPath(beaconRoot).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                        string normFullDir = fullDir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                        if (string.Equals(normBeaconRoot, normFullDir, cmp))
                        {
                            location.PythonExe = beaconPy;
                        }
                    }
                }
            }
            catch
            {
                // Ignore beacon fallback errors during probing
            }
        }

        return true;
    }

    static async Task<FileStream?> TryAcquireStartupLockAsync(int port, TimeSpan timeout)
    {
        string userTag = (Environment.UserName ?? "user");
        foreach (char c in Path.GetInvalidFileNameChars())
        {
            userTag = userTag.Replace(c, '_');
        }
        string lockFilePath = Path.Combine(Path.GetTempPath(), $"hifisampler-client-startup-{port}-{userTag}.lock");
        var sw = Stopwatch.StartNew();
        while (sw.Elapsed < timeout)
        {
            try
            {
                return new FileStream(lockFilePath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
            }
            catch (IOException)
            {
                await Task.Delay(150);
            }
            catch (UnauthorizedAccessException)
            {
                await Task.Delay(150);
            }
        }
        return null;
    }

    static string BuildJsonPayload(string[] args)
    {
        var sb = new StringBuilder(256);
        sb.Append("{\"args\":[");
        for (int i = 0; i < args.Length; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append('"');
            string s = args[i] ?? "";
            foreach (char c in s)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\b': sb.Append("\\b"); break;
                    case '\f': sb.Append("\\f"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        if (char.IsControl(c))
                            sb.Append($"\\u{(int)c:x4}");
                        else
                            sb.Append(c);
                        break;
                }
            }
            sb.Append('"');
        }
        sb.Append("]}");
        return sb.ToString();
    }

    static async Task CommunicateWithServer(string jsonPayload)
    {
        int maxRetries = 2;
        int retryDelayMs = 800;
        bool requestSucceeded = false;

        for (int attempt = 1; attempt <= maxRetries + 1; attempt++)
        {
            try
            {
                using var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");
                Console.WriteLine($"[Communicate] Sending POST (Attempt {attempt}/{maxRetries + 1})...");
                using HttpResponseMessage response = await PostClient.PostAsync($"http://127.0.0.1:{TargetPort}/", content);
                if (response.IsSuccessStatusCode)
                {
                    Console.WriteLine($"[Communicate] Success (Attempt {attempt}, Status: {(int)response.StatusCode})");
                    requestSucceeded = true;
                    break;
                }
                else
                {
                    string responseBody = response.Content == null ? "[No Body]" : await response.Content.ReadAsStringAsync();
                    Console.Error.WriteLine($"[Communicate] Error (Attempt {attempt}): Status {(int)response.StatusCode} {response.ReasonPhrase} - {responseBody}");
                    if (response.StatusCode == HttpStatusCode.ServiceUnavailable && attempt <= maxRetries)
                    {
                        Console.WriteLine("[Communicate] Server still initializing (503), retrying...");
                    }
                    else
                    {
                        break;
                    }
                }
            }
            catch (HttpRequestException ex)
            {
                Console.Error.WriteLine($"[Communicate] Network Error (Attempt {attempt}): {ex.Message}");
            }
            catch (TaskCanceledException)
            {
                Console.Error.WriteLine($"[Communicate] Request Timed Out (Attempt {attempt}).");
                break;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[Communicate] Unexpected Error (Attempt {attempt}): {ex.GetType().Name} - {ex.Message}");
                break;
            }

            if (!requestSucceeded && attempt <= maxRetries)
            {
                await Task.Delay(retryDelayMs);
            }
        }

        if (!requestSucceeded)
        {
            Console.Error.WriteLine("[Communicate] Failed after all attempts.");
            Environment.Exit(1);
        }
    }

    static async Task<bool> IsServerReady(int port)
    {
        try
        {
            using HttpResponseMessage response = await CheckClient.GetAsync($"http://127.0.0.1:{port}/");
            return response.StatusCode == HttpStatusCode.OK;
        }
        catch
        {
            return false;
        }
    }

    static async Task<bool> WaitForServerToStart(int port, int timeoutSeconds, string logPrefix = "[WaitForServer]")
    {
        Console.WriteLine($"{logPrefix} Waiting up to {timeoutSeconds}s for server at port {port}...");
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(timeoutSeconds));
        var sw = Stopwatch.StartNew();
        while (!cts.IsCancellationRequested)
        {
            if (await IsServerReady(port))
            {
                Console.WriteLine($"{logPrefix} Server became ready after {sw.Elapsed.TotalSeconds:F1}s.");
                return true;
            }
            try
            {
                await Task.Delay(TimeSpan.FromMilliseconds(400), cts.Token);
            }
            catch (TaskCanceledException)
            {
                break;
            }
        }
        Console.WriteLine($"{logPrefix} Server did not become ready within {timeoutSeconds}s.");
        return false;
    }

    static bool IsPortInUse(int port)
    {
        const int timeoutMs = 150;
        try
        {
            using var socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            var result = socket.BeginConnect("127.0.0.1", port, null, null);
            if (result.AsyncWaitHandle.WaitOne(timeoutMs))
            {
                socket.EndConnect(result);
                return true;
            }
            return false;
        }
        catch
        {
            return false;
        }
    }

    static void LaunchServerProcess(ServerLocation loc)
    {
        string workingDir = loc.ServerRoot;
        bool hasDirectVenv = !string.IsNullOrEmpty(loc.PythonExe) && File.Exists(loc.PythonExe)
                             && !string.IsNullOrEmpty(loc.ScriptPath) && File.Exists(loc.ScriptPath);

        if (OperatingSystem.IsWindows())
        {
            string innerCmd = hasDirectVenv
                ? $"\"\"{loc.PythonExe}\" \"{loc.ScriptPath}\"\""
                : $"\"\"{loc.LauncherScript}\"\"";

            string arguments =
                $"/C start \"HifiSampler Server\" /D \"{workingDir}\" cmd /C \"title HifiSampler Server & echo Starting HifiSampler Server... & {innerCmd} & echo. & echo Server process finished. & pause\"";

            using var winProc = Process.Start(new ProcessStartInfo
            {
                FileName = "cmd.exe",
                Arguments = arguments,
                WorkingDirectory = workingDir,
                UseShellExecute = false,
                CreateNoWindow = true,
            });
            Console.WriteLine($"[LaunchServer] Started Windows server console in '{workingDir}'.");
            return;
        }

        if (OperatingSystem.IsLinux() || OperatingSystem.IsMacOS())
        {
            string targetExec = hasDirectVenv
                ? $"{QuotePosixSingle(loc.PythonExe)} {QuotePosixSingle(loc.ScriptPath)}"
                : $"bash {QuotePosixSingle(loc.LauncherScript)}";

            string interactiveCmd =
                $"cd {QuotePosixSingle(workingDir)} && echo 'Starting HifiSampler Server...' && {targetExec} ; echo ; echo 'Server exited. Press Enter to close.' ; read -r _";

            bool hasGuiSession = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DISPLAY"))
                                 || !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("WAYLAND_DISPLAY"));

            if (hasGuiSession)
            {
                var terminals = new (string Name, string[] PrefixArgs)[]
                {
                    ("ptyxis", new[] { "--", "bash", "-c", interactiveCmd }),
                    ("kgx", new[] { "--", "bash", "-c", interactiveCmd }),
                    ("gnome-terminal", new[] { "--", "bash", "-c", interactiveCmd }),
                    ("konsole", new[] { "-e", "bash", "-c", interactiveCmd }),
                    ("kitty", new[] { "bash", "-c", interactiveCmd }),
                    ("alacritty", new[] { "-e", "bash", "-c", interactiveCmd }),
                    ("wezterm", new[] { "start", "--", "bash", "-c", interactiveCmd }),
                    ("foot", new[] { "bash", "-c", interactiveCmd }),
                    ("xfce4-terminal", new[] { "-x", "bash", "-c", interactiveCmd }),
                    ("mate-terminal", new[] { "-x", "bash", "-c", interactiveCmd }),
                    ("terminator", new[] { "-x", "bash", "-c", interactiveCmd }),
                    ("lxterminal", new[] { "-e", "bash", "-c", interactiveCmd }),
                    ("xterm", new[] { "-e", "bash", "-c", interactiveCmd }),
                };

                foreach (var term in terminals)
                {
                    if (CommandExistsPosix(term.Name))
                    {
                        try
                        {
                            var psi = new ProcessStartInfo
                            {
                                FileName = CommandExistsPosix("setsid") ? "setsid" : term.Name,
                                WorkingDirectory = workingDir,
                                UseShellExecute = false,
                                CreateNoWindow = false,
                            };
                            if (psi.FileName == "setsid")
                            {
                                psi.ArgumentList.Add(term.Name);
                            }
                            foreach (string arg in term.PrefixArgs)
                            {
                                psi.ArgumentList.Add(arg);
                            }
                            using var termProc = Process.Start(psi);
                            Console.WriteLine($"[LaunchServer] Launched server in detached GUI terminal: {term.Name}");
                            return;
                        }
                        catch (Exception ex)
                        {
                            Console.Error.WriteLine($"[LaunchServer] Terminal '{term.Name}' failed ({ex.Message}), trying next...");
                        }
                    }
                }
            }

            // Fallback: Detached background daemon logging to logs/hifiserver.log
            string logDir = Path.Combine(workingDir, "logs");
            Directory.CreateDirectory(logDir);
            string logFile = Path.Combine(logDir, "hifiserver.log");
            string bgCmd = $"cd {QuotePosixSingle(workingDir)} && exec {targetExec} >> {QuotePosixSingle(logFile)} 2>&1 < /dev/null";

            var bgPsi = new ProcessStartInfo
            {
                FileName = CommandExistsPosix("setsid") ? "setsid" : "nohup",
                WorkingDirectory = workingDir,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            bgPsi.ArgumentList.Add("bash");
            bgPsi.ArgumentList.Add("-c");
            bgPsi.ArgumentList.Add(bgCmd);
            using var bgProc = Process.Start(bgPsi);
            Console.WriteLine($"[LaunchServer] Launched detached background server (log: {logFile}).");
            return;
        }

        throw new PlatformNotSupportedException("Unsupported operating system.");
    }

    static string QuotePosixSingle(string value)
    {
        return "'" + (value ?? "").Replace("'", "'\\''") + "'";
    }

    static bool CommandExistsPosix(string commandName)
    {
        if (string.IsNullOrWhiteSpace(commandName))
        {
            return false;
        }
        if (commandName.Contains('/'))
        {
            return File.Exists(commandName);
        }
        string? pathEnv = Environment.GetEnvironmentVariable("PATH");
        if (string.IsNullOrWhiteSpace(pathEnv))
        {
            return false;
        }
        foreach (string dir in pathEnv.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            try
            {
                string candidate = Path.Combine(dir, commandName);
                if (File.Exists(candidate))
                {
                    return true;
                }
            }
            catch
            {
                // Ignore invalid PATH entries
            }
        }
        return false;
    }
}
