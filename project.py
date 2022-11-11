'''
    Se sabe que el supermercado del cual es gerente cuenta con N cajas, además que
    usted sabe que la cantidad de clientes que se atienden se pueden modelar con
    un proceso de Poisson con λ1 > 0(cantidad promedio de clientes por hora).
    Al llegar, un cliente se forma en la fila en una caja, la cual suponga que las 
    eleccion al buscar aquella que tenga el menor número de personas en fila y si 
    en caso hay empate en la cantidad menor de personas en más de una caja, la cual 
    en el cliente se forma será seleccionada aleatoriamente.Usted sabe que cada 
    cajero despacha al cliente en un tiempo que tiene una distribución exponencial 
    con parámetro λ2 > 0 clientes por hora.
'''
class Queue:
    # La cola se va guardando  [..., tercero, segundo, primero]
    def __init__(self):
        self.items = []

    def is_empty(self):
        # Devuelve True, si la cola está vacía y False en caso contrario
        return self.items == []

    def enqueue(self, item):
       # Agrega el elemento al final de la cola.
        self.items.insert(0, item)

    def dequeue(self):
        # Elimina y devuelve el primer elemento de la cola.
        return self.items.pop()

    def size(self):
        # Retorna el tamaño de la cola
        return len(self.items)

    def __str__(self):
        # Devuelve la cadena de la cola
        return f'{self.items}'


class ReversedQueue:
    # [primero, segundo, tercero, ...] 
    def __init__(self):
        self.items = []

    def is_empty(self):
        # Devuelve True, si la cola está vacía y False en caso contrario
        return self.items == []

    def enqueue(self, item):
        # Agrega el elemento al final de la cola
        self.items.append(item)

    def dequeue(self):
        # Elimina y devuelve el primer elemento de la cola
        return self.items.pop(0)

    def size(self):
        # Devuelve el tamaño de la cola
        return len(self.items)

    def __str__(self):
        # Devuelve la cadena de la cola
        return f'{self.items}'

# 1.Calcule el tiempo promedio de un cliente en cola(tiempo de espera)
# 2.Calcule el número de cliente en la cola
# 3.Calcule el grado de utilización de cada cajero
## a.Para este punto considere los clientes atendidos por cada cajero dividido el número de clientes total