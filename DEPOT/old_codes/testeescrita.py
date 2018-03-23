arq = open('lista.txt', 'w')
texto = []
texto.append('Lista de Alunos\n')
texto.append('---\n')
texto.append('Joao da Silva\n')
texto.append('Jose Lima\n')
texto.append('Maria das Dores')
arq.writelines(texto)
arq.close()