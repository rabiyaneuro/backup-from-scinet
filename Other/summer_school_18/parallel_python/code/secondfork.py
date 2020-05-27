# secondfork.py
import os

while (True):
  pid = os.fork()
  if pid == 0:
    os.execlp("python", "python", "child.py")
    assert False, "Error starting program"
  else:
    print("The child is", pid)
    if input() == "q": break

