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
    static readonly HttpClient PostClient = new HttpClient() { Timeout = TimeSpan.FromSeconds(180) };
    static readonly Func<HttpClient> CheckClientFactory = () => new HttpClient() { Timeout = TimeSpan.FromSeconds(2) };

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

        var location = ResolveServerLocation();

        bool proceedToCommunication = false;
        bool errorOccurred = false;
        string errorMessage = "";

        // --- Phase 1: Quick Readiness Check (No Lock) ---
        if (await IsServerReady(TargetPort))
        {
            proceedToCommunication = true;
        }
        else
        {
            Console.WriteLine("[Main] Server not ready. Acquiring startup lock...");

            // --- Phase 2: Cross-Platform File Lock and Double-Check ---
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
                                // Wait up to 45s (staying safely under OpenUtau's 60s process kill timeout)
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
            proceedToCommunication = true;
        }

        if (proceedToCommunication)
        {
            await CommunicateWithServer(args);
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

    static async Task CommunicateWithServer(string[] args)
    {
        int maxRetries = 2;
        int retryDelayMs = 800;
        bool requestSucceeded = false;
        string jsonPayload = BuildJsonPayload(args);

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
                    // Only retry on 503 ServiceUnavailable (do not retry on 400 Bad Request, 404 Not Found, or 500 Internal Error)
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
            using HttpClient checkClient = CheckClientFactory();
            using HttpResponseMessage response = await checkClient.GetAsync($"http://127.0.0.1:{port}/");
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

            Process.Start(new ProcessStartInfo
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
                            Process.Start(psi);
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
            Process.Start(bgPsi);
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
        try
        {
            var psi = new ProcessStartInfo("sh")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            psi.ArgumentList.Add("-c");
            psi.ArgumentList.Add($"command -v \"{commandName}\" >/dev/null 2>&1");
            using var proc = Process.Start(psi);
            proc?.WaitForExit(500);
            return proc?.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }
}
