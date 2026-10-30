using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.ComponentModel;
using System.Runtime.InteropServices;

[assembly: AssemblyTitle("Drama-3 T8 Launcher")]
[assembly: AssemblyVersion("0.3.2.0")]

internal static class StartDrama3
{
    private static readonly object LogGate = new object();
    private static string logPath;
    private static string statePath;
    private static int port = 7863;
    private static ProcessFamily family;
    private static int existingPid;
    private static long existingStarted;
    private static string existingPython;
    private delegate bool ConsoleHandler(uint signal);
    private static readonly ConsoleHandler CloseHandler = OnConsoleClose;
    [DllImport("kernel32.dll")] private static extern bool SetConsoleCtrlHandler(ConsoleHandler handler, bool add);
    [DllImport("kernel32.dll")] private static extern IntPtr GetConsoleWindow();
    [DllImport("kernel32.dll", SetLastError=true)] private static extern bool AllocConsole();
    [DllImport("kernel32.dll")] private static extern bool FreeConsole();
    [DllImport("user32.dll")] private static extern bool IsWindowVisible(IntPtr window);
    [DllImport("user32.dll")] private static extern bool ShowWindow(IntPtr window, int command);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] private static extern int GetClassName(IntPtr window, StringBuilder name, int maximum);

    private static bool EnsureVisibleConsole(string[] arguments)
    {
        IntPtr window = GetConsoleWindow();
        StringBuilder kind = new StringBuilder(128);
        if (window != IntPtr.Zero) GetClassName(window, kind, kind.Capacity);
        if (window == IntPtr.Zero || kind.ToString() != "ConsoleWindowClass")
        {
            // A pseudoconsole's HWND is only a message window, not a visible
            // terminal. Start a classic host before creating the owned job.
            List<string> command = new List<string>();
            command.Add(Quote(Assembly.GetExecutingAssembly().Location));
            foreach (string argument in arguments) command.Add(Quote(argument));
            ProcessStartInfo host = new ProcessStartInfo(Path.Combine(Environment.SystemDirectory, "conhost.exe"), String.Join(" ", command.ToArray()));
            host.UseShellExecute = true; host.WindowStyle = ProcessWindowStyle.Normal;
            using (Process process = Process.Start(host)) { if (process == null) throw new IOException("无法创建后台控制窗口。"); }
            return false;
        }
        ShowWindow(window, 5);
        // The first call can inherit SW_HIDE from the caller's STARTUPINFO.
        ShowWindow(window, 9);
        return true;
    }

    // The owner joins before starting Python, so descendants inherit membership
    // without a spawn/assignment race. Only this process owns the job handle.
    private sealed class ProcessFamily : IDisposable
    {
        [StructLayout(LayoutKind.Sequential)] private struct BasicLimits
        {
            public long ProcessTime, JobTime;
            public uint Flags;
            public UIntPtr MinimumWorkingSet, MaximumWorkingSet;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint Priority, Scheduling;
        }
        [StructLayout(LayoutKind.Sequential)] private struct Limits
        {
            public BasicLimits Basic;
            public ulong ReadOperations, WriteOperations, OtherOperations, ReadBytes, WriteBytes, OtherBytes;
            public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
        }
        [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
        [DllImport("kernel32.dll", SetLastError=true)] private static extern bool SetInformationJobObject(IntPtr job, int kind, ref Limits limits, uint size);
        [DllImport("kernel32.dll", SetLastError=true)] private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
        [DllImport("kernel32.dll", SetLastError=true)] private static extern bool QueryInformationJobObject(IntPtr job, int kind, IntPtr buffer, uint length, out uint returned);
        [DllImport("kernel32.dll", SetLastError=true)] private static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool member);
        [DllImport("kernel32.dll")] private static extern bool TerminateJobObject(IntPtr job, uint code);
        [DllImport("kernel32.dll")] private static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
        [DllImport("kernel32.dll")] private static extern bool TerminateProcess(IntPtr process, uint code);
        [DllImport("kernel32.dll")] private static extern bool CloseHandle(IntPtr handle);
        private IntPtr handle;
        public ProcessFamily()
        {
            handle = CreateJobObject(IntPtr.Zero, null);
            if (handle == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
            SetKillOnClose(true);
            using (Process owner = Process.GetCurrentProcess())
            {
                if (!AssignProcessToJobObject(handle, owner.Handle))
                {
                    int error = Marshal.GetLastWin32Error(); CloseHandle(handle); handle = IntPtr.Zero;
                    throw new Win32Exception(error, "无法建立子进程回收保护，已停止启动。");
                }
            }
        }
        private void SetKillOnClose(bool enabled)
        {
            Limits limits = new Limits(); limits.Basic.Flags = enabled ? 0x2000u : 0u;
            if (!SetInformationJobObject(handle, 9, ref limits, (uint)Marshal.SizeOf(typeof(Limits))))
                throw new Win32Exception(Marshal.GetLastWin32Error());
        }
        public void Stop() { if (handle != IntPtr.Zero) TerminateJobObject(handle, 0); }
        public void Reap()
        {
            if (handle == IntPtr.Zero) return;
            // Normal app exit also reaps helpers before preserving the exit code.
            int owner = Process.GetCurrentProcess().Id;
            IntPtr buffer = Marshal.AllocHGlobal(1024 * 1024);
            try
            {
                for (int attempt=0; attempt<10; attempt++)
                {
                    uint returned;
                    if (!QueryInformationJobObject(handle, 3, buffer, 1024 * 1024, out returned)) { Stop(); return; }
                    int count = Marshal.ReadInt32(buffer, 4), children = 0;
                    for (int i=0; i<count; i++)
                    {
                        int pid = Marshal.ReadIntPtr(buffer, 8 + i * IntPtr.Size).ToInt32();
                        if (pid == owner) continue;
                        IntPtr process = OpenProcess(1u | 0x100000u | 0x1000u, false, pid);
                        if (process == IntPtr.Zero) continue;
                        try { bool member; if (IsProcessInJob(process, handle, out member) && member) { children++; TerminateProcess(process, 0); } }
                        finally { CloseHandle(process); }
                    }
                    if (children == 0) return;
                    Thread.Sleep(20);
                }
                Stop();
            }
            finally { Marshal.FreeHGlobal(buffer); }
        }
        public void Dispose()
        {
            if (handle == IntPtr.Zero) return;
            Reap(); SetKillOnClose(false); CloseHandle(handle); handle = IntPtr.Zero;
        }
    }

    private static bool OnConsoleClose(uint signal)
    {
        if (signal != 0 && signal != 1 && signal != 2 && signal != 5 && signal != 6) return false;
        if (family != null) { family.Stop(); return true; }
        if (existingPid != 0)
        {
            // A second interactive window may control only the verified service
            // from this exact package, never an arbitrary process on the port.
            try
            {
                using (Process service = Process.GetProcessById(existingPid))
                {
                    if (service.StartTime.ToUniversalTime().Ticks == existingStarted &&
                        String.Equals(service.MainModule.FileName, existingPython, StringComparison.OrdinalIgnoreCase)) service.Kill();
                }
                string launcher = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Start-Drama3.exe");
                int self = Process.GetCurrentProcess().Id;
                foreach (Process owner in Process.GetProcessesByName("Start-Drama3"))
                {
                    using (owner) try { if (owner.Id != self && String.Equals(owner.MainModule.FileName, launcher, StringComparison.OrdinalIgnoreCase)) owner.Kill(); }
                    catch (Exception) { }
                }
            }
            catch (Exception) { }
        }
        return false;
    }

    private static void Log(string message, bool error)
    {
        if (error) Console.Error.WriteLine(message); else Console.WriteLine(message);
        try
        {
            lock (LogGate) File.AppendAllText(logPath,
                DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss ") + message + Environment.NewLine,
                new UTF8Encoding(false));
        }
        catch (Exception) { } // A read-only log folder must not prevent startup.
    }

    private static int Failure(int code, string message)
    {
        Log(message, true);
        if (!String.IsNullOrEmpty(logPath)) Console.Error.WriteLine("启动日志：" + logPath);
        try
        {
            if (Environment.UserInteractive && !Console.IsInputRedirected && !Console.IsOutputRedirected)
            {
                Console.Error.WriteLine("请查看上方错误。按回车关闭此窗口……");
                Console.ReadLine();
            }
        }
        catch (IOException) { }
        catch (InvalidOperationException) { }
        return code;
    }

    private static bool Existing(string python, bool noBrowser)
    {
        try
        {
            if (!File.Exists(statePath)) return false;
            string[] state = File.ReadAllLines(statePath);
            int pid, savedPort; long started;
            if (state.Length != 3 || !Int32.TryParse(state[0], out pid) ||
                !Int64.TryParse(state[1], out started) || !Int32.TryParse(state[2], out savedPort) ||
                savedPort < 1 || savedPort > 65535) return false;
            using (Process process = Process.GetProcessById(pid))
            {
                if (process.HasExited || process.StartTime.ToUniversalTime().Ticks != started ||
                    !String.Equals(Path.GetFullPath(process.MainModule.FileName), Path.GetFullPath(python), StringComparison.OrdinalIgnoreCase)) return false;
            }
            string url = "http://127.0.0.1:" + savedPort + "/";
            HttpWebRequest request = (HttpWebRequest)WebRequest.Create(url + "api/status");
            request.Proxy = null; request.Timeout = 1500; request.ReadWriteTimeout = 1500;
            request.AllowAutoRedirect = false;
            string body;
            using (WebResponse response = request.GetResponse())
            using (StreamReader stream = new StreamReader(response.GetResponseStream(), Encoding.UTF8)) body = stream.ReadToEnd();
            if (!Regex.IsMatch(body, "\"application\"\\s*:\\s*\"drama3-t8\"")) return false;
            Log("工作台已在运行：" + url, false);
            if (!noBrowser)
            {
                ProcessStartInfo open = new ProcessStartInfo(url); open.UseShellExecute = true;
                try
                {
                    using (Process browser = Process.Start(open)) { }
                    Log("已请求浏览器打开现有工作台，无需重复启动模型进程。", false);
                }
                catch (Exception error) { Failure(4, "工作台正在运行，但浏览器未能自动打开。请手动访问 " + url + "\n" + error.Message); }
            }
            if (!noBrowser && !Console.IsOutputRedirected && !Console.IsInputRedirected)
            {
                existingPid = pid; existingStarted = started; existingPython = python;
                Log("此窗口正在控制现有工作台。关闭窗口即可停止服务并释放内存。", false);
                try { using (Process service = Process.GetProcessById(pid)) service.WaitForExit(); }
                finally { existingPid = 0; }
            }
            return true;
        }
        catch (Exception error)
        {
            Log("现有工作台检查：" + error.Message, false);
            return false;
        }
    }
    // Windows/CRT command-line quoting, including quotes and trailing slashes.
    private static string Quote(string argument)
    {
        StringBuilder result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char item in argument)
        {
            if (item == '\\') { slashes++; continue; }
            if (item == '"')
            {
                result.Append('\\', slashes * 2 + 1);
                result.Append('"');
            }
            else
            {
                result.Append('\\', slashes);
                result.Append(item);
            }
            slashes = 0;
        }
        result.Append('\\', slashes * 2);
        result.Append('"');
        return result.ToString();
    }

    private static int Run(string python, string root, IEnumerable<string> arguments, bool appProcess)
    {
        List<string> quoted = new List<string>();
        foreach (string argument in arguments) quoted.Add(Quote(argument));
        ProcessStartInfo info = new ProcessStartInfo();
        info.FileName = python;
        info.Arguments = String.Join(" ", quoted.ToArray());
        info.WorkingDirectory = root;
        info.UseShellExecute = false;
        info.CreateNoWindow = false;
        info.RedirectStandardOutput = true;
        info.RedirectStandardError = true;
        info.StandardOutputEncoding = Encoding.UTF8;
        info.StandardErrorEncoding = Encoding.UTF8;
        using (Process child = Process.Start(info))
        {
            if (child == null) throw new IOException("无法启动整合包内的 Python。");
            child.OutputDataReceived += delegate(object sender, DataReceivedEventArgs args) { if (args.Data != null) Log(args.Data, false); };
            child.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs args) { if (args.Data != null) Log(args.Data, true); };
            child.BeginOutputReadLine(); child.BeginErrorReadLine();
            if (appProcess)
            {
                try { File.WriteAllLines(statePath, new string[] { child.Id.ToString(), child.StartTime.ToUniversalTime().Ticks.ToString(), port.ToString() }, new UTF8Encoding(false)); }
                catch (Exception error) { Log("无法保存启动状态：" + error.Message, false); }
                Log("工作台地址：http://127.0.0.1:" + port + "/", false);
            }
            // A parameterless wait can hang on inherited stdout handles even
            // after the app exits. Process disposal closes the async readers.
            while (!child.WaitForExit(250)) { }
            return child.ExitCode;
        }
    }

    private static string MutexName(string root)
    {
        string identity = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar,
            Path.AltDirectorySeparatorChar).ToUpperInvariant();
        using (SHA256 sha = SHA256.Create())
        {
            byte[] hash = sha.ComputeHash(Encoding.UTF8.GetBytes(identity));
            return "Local\\Drama3T8-" + BitConverter.ToString(hash).Replace("-", "");
        }
    }

    private static bool Check(string root, string python, string updater, string app)
    {
        bool valid = true;
        foreach (string file in new string[] { python, updater, app })
        {
            if (!File.Exists(file))
            {
                Console.Error.WriteLine("缺少文件：" + file);
                valid = false;
            }
        }
        if (valid) Console.WriteLine("DRAMA3_LAUNCHER_CHECK: OK · " + root);
        else Console.Error.WriteLine("请将启动程序放在完整整合包根目录，保留 runtime、tools 和 app.py。");
        return valid;
    }

    private static int Main(string[] arguments)
    {
        bool check = arguments.Length == 1 && arguments[0] == "--check";
        if (!check && Environment.GetEnvironmentVariable("DRAMA_LAUNCHER_HEADLESS") != "1")
            if (!EnsureVisibleConsole(arguments)) return 0;
        try { Console.OutputEncoding = new UTF8Encoding(false); }
        catch (IOException) { }
        try { Console.Title = "Drama-3 T8 · 关闭窗口停止服务"; } catch (Exception) { }
        SetConsoleCtrlHandler(CloseHandler, true);
        string root = Path.GetFullPath(AppDomain.CurrentDomain.BaseDirectory);
        string python = Path.Combine(root, "runtime", "python", "python.exe");
        string updater = Path.Combine(root, "tools", "update_portable.py");
        string app = Path.Combine(root, "app.py");
        statePath = Path.Combine(root, "userdata", "launcher-state.txt");
        try
        {
            if (!check)
            {
                string logs = Path.Combine(root, "userdata", "logs"); Directory.CreateDirectory(logs);
                logPath = Path.Combine(logs, "launcher-" + DateTime.Now.ToString("yyyyMMdd") + ".log");
            }
        }
        catch (Exception) { }
        bool noBrowser = Array.IndexOf(arguments, "--no-browser") >= 0;
        for (int i = 0; i < arguments.Length; i++)
        {
            int candidate;
            string value = arguments[i] == "--port" && i + 1 < arguments.Length ? arguments[i+1] :
                arguments[i].StartsWith("--port=") ? arguments[i].Substring(7) : "";
            if (Int32.TryParse(value, out candidate) && candidate > 0 && candidate <= 65535) port = candidate;
        }
        foreach (string argument in arguments)
        {
            if (argument == "--check" && !check)
            {
                return Failure(2, "--check 必须单独使用；此模式不启动 Python 或网络检查。");
            }
        }
        try
        {
            if (!Check(root, python, updater, app)) return Failure(2, "整合包必需文件缺失，已停止启动。");
            if (check) return 0;
            if (Existing(python, noBrowser)) return 0;
            using (Mutex mutex = new Mutex(false, MutexName(root)))
            {
                bool owned = false;
                try
                {
                    try { owned = mutex.WaitOne(0, false); }
                    catch (AbandonedMutexException) { owned = true; }
                    if (!owned)
                    {
                        Log("此目录的 Drama-3 正在启动，正在检查工作台地址……", false);
                        if (noBrowser) return 0;
                        for (int attempt = 0; attempt < 30; attempt++)
                        {
                            if (Existing(python, false)) return 0;
                            Thread.Sleep(500);
                        }
                        return Failure(3, "已有启动进程尚未提供工作台页面，请检查启动日志；没有重复启动 Python。");
                    }
                    Log("Drama-3 T8 · " + root, false);
                    family = new ProcessFamily();
                    Log("保持此窗口运行。关闭窗口将停止 Studio、转写及子进程，释放模型内存。", false);
                    int updated = Run(python, root, new string[] { "-X", "utf8", updater, "--automatic" }, false);
                    if (updated == 75) return 0; // Same stop-without-launch behavior as Start-Drama3.cmd.
                    if (updated != 0)
                    {
                        return Failure(updated, "更新或恢复未完成，已停止启动。退出码：" + updated);
                    }
                    List<string> start = new List<string>(new string[] { "-X", "utf8", app });
                    start.AddRange(arguments);
                    int finished = Run(python, root, start, true);
                    if (finished != 0) return Failure(finished, "Drama-3 已退出。退出码：" + finished);
                    return finished;
                }
                finally { if (family != null) { family.Dispose(); family = null; } if (owned) mutex.ReleaseMutex(); }
            }
        }
        catch (Exception error)
        {
            return Failure(1, "Drama-3 启动失败：" + error.Message + "\n整合包目录：" + root);
        }
    }
}
