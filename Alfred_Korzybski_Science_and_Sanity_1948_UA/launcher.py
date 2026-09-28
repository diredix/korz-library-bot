
import subprocess
log = r"C:/Users/diredix/Downloads/korz_71_84_hq/db_install3.log"
with open(log,"w") as f: f.write("start\n")
p = subprocess.Popen("dev-browser install", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
with open(r"C:/Users/diredix/Downloads/korz_71_84_hq/db_pid.txt","w") as f: f.write(str(p.pid))
data = p.communicate()[0]
with open(log,"ab") as f: f.write(data)
with open(log,"a") as f: f.write("\nDONE rc=%s\n"%p.returncode)
