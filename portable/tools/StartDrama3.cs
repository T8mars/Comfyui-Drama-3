using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading;

[assembly: AssemblyTitle("Drama-3 T8 Launcher")]
[assembly: AssemblyVersion("0.3.0.0")]

internal static class StartDrama3
{
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

    private static int Run(string python, string root, IEnumerable<string> arguments)
    {
        List<string> quoted = new List<string>();
        foreach (string argument in arguments) quoted.Add(Quote(argument));
        ProcessStartInfo info = new ProcessStartInfo();
        info.FileName = python;
        info.Arguments = String.Join(" ", quoted.ToArray());
        info.WorkingDirectory = root;
        info.UseShellExecute = false;
        info.CreateNoWindow = false;
        using (Process child = Process.Start(info))
        {
            if (child == null) throw new IOException("无法启动整合包内的 Python。");
            child.WaitForExit();
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
        try { Console.OutputEncoding = new UTF8Encoding(false); }
        catch (IOException) { }
        string root = Path.GetFullPath(AppDomain.CurrentDomain.BaseDirectory);
        string python = Path.Combine(root, "runtime", "python", "python.exe");
        string updater = Path.Combine(root, "tools", "update_portable.py");
        string app = Path.Combine(root, "app.py");
        bool check = arguments.Length == 1 && arguments[0] == "--check";
        foreach (string argument in arguments)
        {
            if (argument == "--check" && !check)
            {
                Console.Error.WriteLine("--check 必须单独使用；此模式不启动 Python 或网络检查。");
                return 2;
            }
        }
        try
        {
            if (!Check(root, python, updater, app)) return 2;
            if (check) return 0;
            using (Mutex mutex = new Mutex(false, MutexName(root)))
            {
                bool owned = false;
                try
                {
                    try { owned = mutex.WaitOne(0, false); }
                    catch (AbandonedMutexException) { owned = true; }
                    if (!owned)
                    {
                        Console.WriteLine("此目录的 Drama-3 已在启动或运行，请使用原窗口。");
                        return 0;
                    }
                    Console.WriteLine("Drama-3 T8 · " + root);
                    int updated = Run(python, root, new string[] { "-X", "utf8", updater, "--automatic" });
                    if (updated == 75) return 0; // Same stop-without-launch behavior as Start-Drama3.cmd.
                    if (updated != 0)
                    {
                        Console.Error.WriteLine("更新或恢复未完成，已停止启动。退出码：" + updated);
                        return updated;
                    }
                    List<string> start = new List<string>(new string[] { "-X", "utf8", app });
                    start.AddRange(arguments);
                    int finished = Run(python, root, start);
                    if (finished != 0) Console.Error.WriteLine("Drama-3 已退出。退出码：" + finished);
                    return finished;
                }
                finally { if (owned) mutex.ReleaseMutex(); }
            }
        }
        catch (Exception error)
        {
            Console.Error.WriteLine("Drama-3 启动失败：" + error.Message);
            Console.Error.WriteLine("整合包目录：" + root);
            return 1;
        }
    }
}
