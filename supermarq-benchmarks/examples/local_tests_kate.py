import supermarq
from supermarq.benchmarks.vqe_proxy import VQEProxy

test_vqe = VQEProxy(3,1)
for i in range(len(test_vqe.circuit())):
    print(i)
    print(test_vqe.circuit()[i])
    print()