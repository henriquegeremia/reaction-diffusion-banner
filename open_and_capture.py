import subprocess
import time

url = "C:\\Users\\henri\\my-drive\\projects\\reaction-diffusion-banner\\index.html"

# Run chrome directly
cmd = f'schtasks /create /tn "AGYOpen" /tr "cmd.exe /c start chrome {url}" /sc once /st 23:59 /f'
subprocess.run(cmd, shell=True)
subprocess.run('schtasks /run /tn "AGYOpen"', shell=True)
time.sleep(1)
subprocess.run('schtasks /delete /tn "AGYOpen" /f', shell=True)

time.sleep(3)

# Take screenshot
cmd_shot = 'schtasks /create /tn "AGYScreenshot" /tr "powershell -ExecutionPolicy Bypass -File C:\\Users\\henri\\my-drive\\projects\\reaction-diffusion-banner\\take_shot.ps1" /sc once /st 23:59 /f'
subprocess.run(cmd_shot, shell=True)
subprocess.run('schtasks /run /tn "AGYScreenshot"', shell=True)
time.sleep(2)
subprocess.run('schtasks /delete /tn "AGYScreenshot" /f', shell=True)
print("COMPLETED")
