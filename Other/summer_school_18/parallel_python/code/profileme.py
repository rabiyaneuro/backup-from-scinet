#file: profileme.py
@profile
def profilewrapper():
  x=[1.0]*(2048*2048)  
  a=str(x[0])
  a+="\nis a one\n"
  del x
  print(a)
profilewrapper()
