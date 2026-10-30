"""Real Windows compilation/child-process tests; no Python runtime/model/service."""
from pathlib import Path
import importlib.util
import os
import shutil
import subprocess
import time
import base64
import sys
import json

import pytest

ROOT=Path(__file__).resolve().parents[1]
CSC=Path(os.environ.get('WINDIR','C:/Windows'))/'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
pytestmark=pytest.mark.skipif(os.name!='nt' or not CSC.is_file(),reason='Windows .NET Framework C# compiler required')
FLAGS=getattr(subprocess,'CREATE_NO_WINDOW',0)

STUB=r'''
using System;
using System.IO;
using System.Text;
using System.Threading;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
internal static class Stub {
    static string Encode(string value) { return Convert.ToBase64String(Encoding.UTF8.GetBytes(value)); }
    static int Number(string name) { string value=Environment.GetEnvironmentVariable(name); return value==null?0:Int32.Parse(value); }
    static int Main(string[] args) {
        if(args.Length==1 && args[0]=="--helper") {
            File.WriteAllText(Environment.GetEnvironmentVariable("DRAMA_LAUNCHER_HELPER_PID"),Process.GetCurrentProcess().Id.ToString());
            TcpListener listener=new TcpListener(IPAddress.Loopback,Number("DRAMA_LAUNCHER_HELPER_PORT"));listener.Start();
            Thread.Sleep(30000);listener.Stop();return 0;
        }
        bool update=args.Length>2 && Path.GetFileName(args[2])=="update_portable.py";
        string line=(update?"update":"app")+"\t"+Encode(Environment.CurrentDirectory);
        foreach(string arg in args) line+="\t"+Encode(arg);
        File.AppendAllText(Environment.GetEnvironmentVariable("DRAMA_LAUNCHER_TEST_LOG"),line+"\n",new UTF8Encoding(false));
        if(update)return Number("DRAMA_LAUNCHER_UPDATE_EXIT");
        if(Number("DRAMA_LAUNCHER_SPAWN_HELPER")!=0) {
            ProcessStartInfo helper=new ProcessStartInfo(Process.GetCurrentProcess().MainModule.FileName,"--helper");
            helper.UseShellExecute=false;helper.CreateNoWindow=true;Process.Start(helper);
        }
        Thread.Sleep(Number("DRAMA_LAUNCHER_HOLD_MS"));
        return Number("DRAMA_LAUNCHER_APP_EXIT");
    }
}
'''


@pytest.fixture(scope='module')
def compiled(tmp_path_factory):
    spec=importlib.util.spec_from_file_location('tested_build_launcher',ROOT/'tools/build_launcher.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    root=tmp_path_factory.mktemp('launcher-compiler')
    launcher=helper.build_launcher(root/'Start-Drama3.exe')
    source=root/'stub.cs';source.write_text(STUB,encoding='utf-8')
    stub=root/'python.exe'
    result=subprocess.run([str(CSC),'/nologo','/target:winexe','/platform:x64','/optimize+',
        '/codepage:65001',f'/out:{stub}',str(source)],capture_output=True,text=True,
        encoding='utf-8',errors='replace',creationflags=FLAGS)
    assert result.returncode==0,result.stdout+result.stderr
    return helper,launcher,stub


def package(folder,compiled):
    folder=folder/'整合包 中文 spaces & (portable)'
    (folder/'runtime/python').mkdir(parents=True)
    (folder/'tools').mkdir()
    shutil.copy2(compiled[1],folder/'Start-Drama3.exe')
    shutil.copy2(compiled[2],folder/'runtime/python/python.exe')
    (folder/'tools/update_portable.py').write_text('# no execution: compiled stub records arguments',encoding='utf-8')
    (folder/'app.py').write_text('# no execution: compiled stub records arguments',encoding='utf-8')
    env={**os.environ,'DRAMA_LAUNCHER_TEST_LOG':str(folder/'actual-child-arguments.tsv'),
        'DRAMA_LAUNCHER_UPDATE_EXIT':'0','DRAMA_LAUNCHER_APP_EXIT':'0','DRAMA_LAUNCHER_HOLD_MS':'0',
        'DRAMA_LAUNCHER_HEADLESS':'1'}
    return folder,env


def run(folder,env,*args):
    return subprocess.run([str(folder/'Start-Drama3.exe'),*args],env=env,cwd=folder.parent,
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=10,creationflags=FLAGS)


def events(env):
    path=Path(env['DRAMA_LAUNCHER_TEST_LOG'])
    if not path.exists():return []
    return [[parts[0],*[base64.b64decode(value).decode('utf-8') for value in parts[1:]]]
        for parts in [line.split('\t') for line in path.read_text(encoding='utf-8').splitlines()]]


def test_check_only_reads_required_paths_without_starting_children(tmp_path,compiled):
    folder,env=package(tmp_path,compiled)
    checked=run(folder,env,'--check')
    assert checked.returncode==0 and 'DRAMA3_LAUNCHER_CHECK: OK' in checked.stdout
    assert events(env)==[]


def test_missing_file_check_and_normal_start_do_not_launch_python(tmp_path,compiled):
    folder,env=package(tmp_path,compiled)
    (folder/'app.py').unlink()
    for args in [(' --unused ',),('--check',)]:
        checked=run(folder,env,*args)
        assert checked.returncode==2 and 'app.py' in checked.stderr
    assert events(env)==[]


def test_check_cannot_be_mixed_with_app_arguments(tmp_path,compiled):
    folder,env=package(tmp_path,compiled)
    checked=run(folder,env,'--check','--port','12345')
    assert checked.returncode==2 and '--check' in checked.stderr and events(env)==[]


def test_unicode_shell_characters_quotes_and_backslashes_are_forwarded_bitexact(tmp_path,compiled):
    folder,env=package(tmp_path,compiled)
    marker=folder/'must-not-be-created.txt'
    arguments=['--no-browser','--port','12345','','two words','中文 路径','emoji🎬',
        'literal"quote','a\\"b','space with trailing\\','\\\\server\\share\\',
        'tabs\tinside','a&b|c>d<e^f%g',f'& echo unexpected > "{marker}"']
    result=run(folder,env,*arguments)
    assert result.returncode==0,result.stderr
    recorded=events(env)
    assert len(recorded)==2
    assert recorded[0]==['update',str(folder),'-X','utf8',str(folder/'tools/update_portable.py'),'--automatic']
    assert recorded[1]==['app',str(folder),'-X','utf8',str(folder/'app.py'),*arguments]
    assert not marker.exists()


@pytest.mark.parametrize('code',[2,7,74,76])
def test_any_update_error_blocks_app_and_returns_its_code(tmp_path,compiled,code):
    folder,env=package(tmp_path,compiled);env['DRAMA_LAUNCHER_UPDATE_EXIT']=str(code)
    result=run(folder,env)
    assert result.returncode==code and str(code) in result.stderr
    assert [row[0] for row in events(env)]==['update']


def test_update_busy75_matches_cmd_stop_without_launch_success_exit(tmp_path,compiled):
    folder,env=package(tmp_path,compiled);env['DRAMA_LAUNCHER_UPDATE_EXIT']='75'
    result=run(folder,env)
    assert result.returncode==0 and [row[0] for row in events(env)]==['update']


def test_app_exit_is_propagated_without_an_interactive_pause(tmp_path,compiled):
    folder,env=package(tmp_path,compiled);env['DRAMA_LAUNCHER_APP_EXIT']='17'
    result=run(folder,env)
    assert result.returncode==17 and '17' in result.stderr
    assert [row[0] for row in events(env)]==['update','app']


def test_mutex_blocks_same_directory_for_entire_app_but_not_another_directory(tmp_path,compiled):
    import socket
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    folder,env=package(tmp_path/'first',compiled);env['DRAMA_LAUNCHER_HOLD_MS']='2500'
    first=subprocess.Popen([str(folder/'Start-Drama3.exe'),'--no-browser','--port',str(port)],env=env,cwd=folder.parent,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=FLAGS)
    try:
        deadline=time.monotonic()+5
        while len(events(env))<2 and time.monotonic()<deadline:time.sleep(.02)
        assert len(events(env))==2 and first.poll() is None
        duplicate=run(folder,env,'--no-browser')
        assert duplicate.returncode==0 and len(events(env))==2
        other,other_env=package(tmp_path/'second',compiled)
        distinct=run(other,other_env)
        assert distinct.returncode==0 and len(events(other_env))==2
        assert first.wait(timeout=5)==0
    finally:
        if first.poll() is None:first.wait(timeout=5)
        first.communicate()


def test_running_service_reopens_address_without_starting_any_more_children(tmp_path,compiled):
    import http.server
    import threading
    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path=='/api/status'
            body=b'{"application":"drama3-t8"}'
            self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Health)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    folder,env=package(tmp_path,compiled);env['DRAMA_LAUNCHER_HOLD_MS']='2500'
    first=subprocess.Popen([str(folder/'Start-Drama3.exe'),'--no-browser','--port',str(server.server_port)],env=env,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=FLAGS)
    try:
        deadline=time.monotonic()+5
        while len(events(env))<2 and time.monotonic()<deadline:time.sleep(.02)
        result=run(folder,env,'--no-browser')
        assert result.returncode==0 and '工作台已在运行' in result.stdout
        assert str(server.server_port) in result.stdout and len(events(env))==2 and first.poll() is None
    finally:
        first.wait(timeout=5);first.communicate();server.shutdown();server.server_close()


def test_real_failure_is_durable_and_noninteractive_execution_never_hangs(tmp_path,compiled):
    folder,env=package(tmp_path,compiled);env['DRAMA_LAUNCHER_APP_EXIT']='17'
    result=run(folder,env,'--no-browser')
    assert result.returncode==17 and '启动日志' in result.stderr
    logs=list((folder/'userdata/logs').glob('launcher-*.log'))
    assert len(logs)==1 and '退出码：17' in logs[0].read_text(encoding='utf-8')


def test_failed_compile_preserves_existing_launcher_and_cleans_only_own_temporary(tmp_path,compiled):
    target=tmp_path/'Start-Drama3.exe';shutil.copy2(compiled[1],target)
    before=target.read_bytes();bad=tmp_path/'bad.cs';bad.write_text('this is not C#',encoding='utf-8')
    with pytest.raises(RuntimeError,match='compilation failed'):
        compiled[0].build_launcher(target,source=bad)
    assert target.read_bytes()==before and bad.is_file()
    assert not list(tmp_path.glob('.drama-launcher-*.exe'))


@pytest.mark.parametrize('shutdown',['normal_exit','terminate_owner','close_console','allocate_console'])
def test_owned_app_and_grandchild_release_port_when_launcher_stops(tmp_path,compiled,shutdown):
    import ctypes
    import socket
    import psutil
    folder,env=package(tmp_path,compiled)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    env.update(DRAMA_LAUNCHER_SPAWN_HELPER='1',DRAMA_LAUNCHER_HELPER_PORT=str(port),
        DRAMA_LAUNCHER_HELPER_PID=str(folder/'helper.pid'),
        DRAMA_LAUNCHER_HOLD_MS='1500' if shutdown=='normal_exit' else '30000')
    if shutdown=='allocate_console':env.pop('DRAMA_LAUNCHER_HEADLESS')
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    flags=subprocess.CREATE_NEW_CONSOLE if shutdown=='close_console' else FLAGS
    output=(folder/'owner-output.log').open('wb')
    owner=subprocess.Popen([str(folder/'Start-Drama3.exe'),'--no-browser','--port',str(port)],env=env,
        stdout=output,stderr=output,creationflags=flags,startupinfo=startup)
    tracked=[]
    try:
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            state=folder/'userdata/launcher-state.txt';helper=folder/'helper.pid'
            if state.is_file() and helper.is_file():
                try:
                    tracked=[int(state.read_text('utf-8').splitlines()[0]),int(helper.read_text())]
                    with socket.create_connection(('127.0.0.1',port),timeout=.1):break
                except (ValueError,OSError):pass
            time.sleep(.02)
        else:pytest.fail('Owned helper failed to listen')
        assert all(psutil.pid_exists(pid) for pid in tracked)
        control_pid=psutil.Process(tracked[0]).ppid() if shutdown=='allocate_console' else owner.pid
        if shutdown=='terminate_owner':owner.terminate()
        elif shutdown in {'close_console','allocate_console'}:
            probe='import ctypes,sys,json; k=ctypes.WinDLL("kernel32"); k.GetConsoleWindow.restype=ctypes.c_void_p; k.FreeConsole(); assert k.AttachConsole(int(sys.argv[1])); h=k.GetConsoleWindow(); u=ctypes.WinDLL("user32"); u.IsWindowVisible.argtypes=[ctypes.c_void_p]; print(json.dumps({"hwnd":h,"visible":bool(u.IsWindowVisible(h))}))'
            window=subprocess.run([sys.executable,'-c',probe,str(control_pid)],capture_output=True,text=True,creationflags=FLAGS,timeout=5)
            assert window.returncode==0,window.stderr
            information=json.loads(window.stdout);hwnd=information['hwnd'];assert hwnd
            if shutdown=='allocate_console':assert information['visible'],information
            user=ctypes.WinDLL('user32',use_last_error=True)
            user.PostMessageW.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.c_size_t,ctypes.c_ssize_t]
            assert user.PostMessageW(hwnd,0x10,0,0)
        owner.wait(timeout=8)
        if shutdown=='normal_exit':assert owner.returncode==0
        deadline=time.monotonic()+4
        while any(psutil.pid_exists(pid) for pid in tracked) and time.monotonic()<deadline:time.sleep(.02)
        assert not any(psutil.pid_exists(pid) for pid in tracked)
        with socket.socket() as listener:listener.bind(('127.0.0.1',port))
    finally:
        if owner.poll() is None:owner.terminate();owner.wait(timeout=5)
        for pid in tracked:
            try:
                process=psutil.Process(pid)
                if Path(process.exe()).resolve()==(folder/'runtime/python/python.exe').resolve():process.kill()
            except psutil.NoSuchProcess:pass
        output.close()


def test_cmd_fallback_supervisor_reaps_real_python_app_and_worker(tmp_path):
    import socket
    import psutil
    root=tmp_path/'CMD fallback 中文'
    (root/'drama3_t8').mkdir(parents=True)
    (root/'drama3_t8/__init__.py').write_text('',encoding='utf-8')
    shutil.copy2(ROOT/'drama3_t8/console_launcher.py',root/'drama3_t8/console_launcher.py')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    (root/'worker.py').write_text('import socket,time,sys\ns=socket.socket();s.bind(("127.0.0.1",int(sys.argv[1])));s.listen();time.sleep(30)\n',encoding='utf-8')
    (root/'app.py').write_text('import subprocess,sys,time,os\nfrom pathlib import Path\nPath("app.pid").write_text(str(os.getpid()))\np=subprocess.Popen([sys.executable,"worker.py",sys.argv[1]])\nPath("worker.pid").write_text(str(p.pid))\ntime.sleep(30)\n',encoding='utf-8')
    owner=subprocess.Popen([sys.executable,'-X','utf8','-m','drama3_t8.console_launcher',str(root/'app.py'),str(port)],
        cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,creationflags=FLAGS)
    tracked=[]
    try:
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            if (root/'app.pid').is_file() and (root/'worker.pid').is_file():
                try:
                    tracked=[int((root/name).read_text()) for name in ['app.pid','worker.pid']]
                    with socket.create_connection(('127.0.0.1',port),timeout=.1):break
                except (ValueError,OSError):pass
            time.sleep(.02)
        else:pytest.fail('CMD supervisor worker failed to listen')
        owner.terminate();owner.wait(timeout=5)
        deadline=time.monotonic()+4
        while any(psutil.pid_exists(pid) for pid in tracked) and time.monotonic()<deadline:time.sleep(.02)
        assert not any(psutil.pid_exists(pid) for pid in tracked)
        with socket.socket() as listener:listener.bind(('127.0.0.1',port))
    finally:
        if owner.poll() is None:owner.terminate();owner.wait(timeout=5)
        for pid in tracked:
            try:
                process=psutil.Process(pid)
                if Path(process.exe()).resolve()==Path(sys.executable).resolve():process.kill()
            except psutil.NoSuchProcess:pass
        owner.communicate()
