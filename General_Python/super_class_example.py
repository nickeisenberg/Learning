class parent:

    def __init__(self, text, a, b):
        self.text = text
        self.a = a
        self.b = b

    def print_text(self):
        print(self.text)

    def add_nums(self):
        print(self.a + self.b)

class child(parent):

    def __init__(self, text, a, b, c, d):
        super().__init__(text, a, b)
        self.c = c
        self.d = d

    def subtract_nums(self):
        print(self.c - self.d)

child_instance = child(text='here is the text',
                       a=3,
                       b=7,
                       c=1,
                       d=2)

print(child_instance.a)
print(child_instance.b)
print(child_instance.c)
print(child_instance.d)

child_instance.print_text()
child_instance.add_nums()
child_instance.subtract_nums()


