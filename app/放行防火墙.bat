@echo off
:: ============================================
:: ElevatorGuard 防火墙放行脚本
:: 作用: 放行 8300(HTTP) + 8301(HTTPS摄像头) 端口
:: 用法: 右键本文件 → "以管理员身份运行"
:: ============================================

echo ElevatorGuard 防火墙放行脚本
echo ================================
echo.

:: 放行 HTTP 8300 (页面/检测/批量/拍照)
netsh advfirewall firewall show rule name="ElevatorGuard" >nul 2>&1
if %errorlevel%==0 (
    echo [OK] HTTP 8300 端口规则已存在
) else (
    echo 正在放行 TCP 8300 端口...
    netsh advfirewall firewall add rule name="ElevatorGuard" dir=in action=allow protocol=TCP localport=8300 >nul 2>&1
    if %errorlevel%==0 (
        echo [成功] HTTP 8300 已放行
    ) else (
        echo [失败] 请以管理员身份运行
        goto :end
    )
)

:: 放行 HTTPS 8301 (手机摄像头)
netsh advfirewall firewall show rule name="ElevatorGuardHTTPS" >nul 2>&1
if %errorlevel%==0 (
    echo [OK] HTTPS 8301 端口规则已存在
) else (
    echo 正在放行 TCP 8301 端口(手机摄像头用)...
    netsh advfirewall firewall add rule name="ElevatorGuardHTTPS" dir=in action=allow protocol=TCP localport=8301 >nul 2>&1
    if %errorlevel%==0 (
        echo [成功] HTTPS 8301 已放行
    ) else (
        echo [失败] 请以管理员身份运行
    )
)

echo.
echo ================================
echo 当前电脑局域网 IP:
for /f "tokens=2 delims=:" %%a in ('netsh ip ipv4 show addresses ^| findstr /i "192.168"') do (
    echo.
    echo   页面/检测/批量/拍照:  http://%%a:8300
    echo   手机实时摄像头:       https://%%a:8301
)
echo.
echo 手机需连同一WiFi/热点
echo HTTPS地址首次打开会弹"不安全"提示,点"高级→继续访问"即可
echo.
pause
exit /b 0

:end
echo.
echo 请右键本文件,选择"以管理员身份运行"后重试
pause
