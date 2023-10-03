import torch

class Question1:
    def __init__(self):
        self.a = torch.rand(1, requires_grad=True)
        self.b = torch.rand(1, requires_grad=True)
        self.c = torch.rand(1, requires_grad=True)
        self.d = torch.rand(1, requires_grad=True)

    def forward(self, x):
        return self.a * x**3 + self.b * x**2 + self.c * x + self.d

class Question2:
    def forward(self, x):
        return torch.exp(-0.5 * x) * torch.sin(2 * x)



class Question3:
    def __init__(self):
        self.w = torch.rand(3, 3, requires_grad=True)
        self.b = torch.rand(3, 3, requires_grad=True)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + torch.exp(-z))

    def forward(self, x):
        z = torch.mm(self.w, x) + self.b
        result = self.sigmoid(z)
        return result

    
class Question4:
    def __init__(self):
        self.v = torch.rand(1,3, requires_grad=True)
        self.X = torch.rand(2,3, requires_grad=True)
        self.a = torch.rand(2,1, requires_grad=True)

    def forward(self):
        result = self.v.mm(self.X.T.mm(self.a))
        return result
class Question5:
    def __init__(self):
        self.d = 3
        self.n = 2
        self.v = torch.rand(self.d, 1, requires_grad=True)
        self.X = torch.rand(self.d, self.n, requires_grad=True)
        self.A = torch.rand(self.d, self.d, requires_grad=True)
        self.B = torch.rand(self.d, self.d, requires_grad=True)
        self.z = torch.rand(self.d, 1, requires_grad=True)

    def forward(self):
        result = self.v.t() @ self.A @ self.X @ self.X.t() @ self.B @ self.z
        return result

def custom_reduction(output):
    return torch.mean(output) 
        

def main():
    torch.manual_seed(0)
    x = torch.randn(1, requires_grad=True)
    x_2d = torch.randn(2, requires_grad=True)

    # Question 1
    q1 = Question1()
    output = q1.forward(x)
    output.backward()
    print("Question 1:")
    print("f(x) = a * x^3 + b * x^2 + c * x + d = {output.item()}")
    print(f"Gradient with respect to a: {q1.a.grad.item()}")
    print(f"Gradient with respect to b: {q1.b.grad.item()}")
    print(f"Gradient with respect to c: {q1.c.grad.item()}")
    print(f"Gradient with respect to d: {q1.d.grad.item()}\n")

    # Question 2
    q2 = Question2()
    output = q2.forward(x)
    output.backward()
    print("Question 2:")
    print(f"f(x) = e^(-0.5 * x) * sin(2 * x) = {output.item()}")
    print(f"Gradient with respect to x: {x.grad.item()}\n")

    # Question 3
    q3 = Question3()
    output = q3.forward(torch.rand(3, 1))
    output = custom_reduction(output)
    output.backward()
    print("Question 3:")
    print(f"f(x) = sigmoid(w * x + b) = {output.item()}")
    print(f"Gradient with respect to w:\n{q3.w.grad}")
    print(f"Gradient with respect to b:\n{q3.b.grad}\n")

    # Question 4
    
    q4 = Question4()
    output = q4.forward()
    output = custom_reduction(output)
    output.backward()
    print("Question 4:")
    print(f"f(X) = v * X^a = {output.item()}")
    print(f"Gradient with respect to v:\n{q4.v.grad}")
    print(f"Gradient with respect to X:\n{q4.X.grad}")
    print(f"Gradient with respect to a:\n{q4.a.grad}\n")

    # Question 5
    q5 = Question5()
    output = q5.forward()
    output.backward()
    print(f"f(X) = v.t() @ A @ X @ X.t @ B @ z = {output.item()}")
    print(f"Gradient with respect to X:\n{q5.X.grad}")


if __name__ == "__main__":
    main()