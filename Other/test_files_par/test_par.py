import sys
import pp

ppservers = ()

job_server = pp.Server(5) 
print "Starting pp with", job_server.get_ncpus(), "workers"


print(sys.version)
print("hello world")

def rand_f(num):
	p= 0
	for i in range(num):
		p = i +1
	return p

f1 = job_server.submit(rand_f, (10000000,), (), ("sys",))

r1 = f1()

#inputs = (100000, 100100, 100200, 100300, 100400, 100500, 100600, 100700)
#jobs = [(input, job_server.submit(sum_primes,(input,), (isprime,), ("math",))) for input in inputs]
#for input, job in jobs:
#    print "Sum of primes below", input, "is", job()