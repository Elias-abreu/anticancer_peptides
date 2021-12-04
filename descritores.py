class Descritores:

    def __init__(self):
        self.amiacidos = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "Q", "P", "R", "S", "T", "V", "W",
                          "Y"]

    def composicao_aminoacidos(self, pep):
        vetorAAC = [0] * 20  # Armazena a porcentagem que cada resíduo aparece no peptídeo
        vetorQTD = [0] * 20  # Armazena a quantidade que cada resíduo aparece no peptídeo
        # print(vetorAAC)
        for i in range(len(pep) - 1):
            # print("AQUI ",pep[i])
            posicao_amino = self.amiacidos.index(pep[i])  # Posição do i no vetor de aminoácidos
            vetorQTD[posicao_amino] += 1
            vetorAAC[posicao_amino] = vetorQTD[posicao_amino] / (len(pep))  # Calculando a frequência
        # print(vetorQTD)
        return vetorAAC

        # Ligação entre os grupos, frequência ou quantidade de ligação]

    def depeptideo(self, pep):
        combinacoes_posiveis = []
        vetor_total_ligacao = [0] * 400
        vetor_dipeptideo = [0] * 400
        # Gerar todas as combinações possíveis
        for i in range(len(self.amiacidos)):
            for j in range(len(self.amiacidos)):
                combinacoes_posiveis.append(self.amiacidos[i] + self.amiacidos[j])
        vizinhos = []
        for i in range(len(pep) - 2):
            v = pep[i] + pep[i + 1]
            indice = self.buscarVetorG(combinacoes_posiveis, v)
            vetor_total_ligacao[indice] += 1
            vetor_dipeptideo[indice] = vetor_total_ligacao[indice] / ((len(pep) - 1))
            # print(pep[i]+pep[i+1])
            vizinhos.append(v)

        return vetor_dipeptideo

    def buscarVetorG(self, lista, valor):
        for i in lista:
            if (i == valor):
                return lista.index(i)

    def perfilBinario(self, peptideo, posicao):
        vetN = [0] * 20
        vetC = [0] * 20
        if (len(peptideo) > posicao + 1):
            aminoN = peptideo[posicao]
            aminoC = peptideo[len(peptideo) - posicao - 2]
            # print(peptideo)
            # print(aminoN," ",aminoC)
            pN = self.buscarVetorG(self.amiacidos, aminoN)
            vetN[pN] = 1
            pN = self.buscarVetorG(self.amiacidos, aminoC)
            vetC[pN] = 1
        return vetN, vetC

    # 5,10,15
    def perfilBinario2(self, peptideo, posicao):
        recursosN = []
        recursosC = []
        contador = 0
        while contador <= posicao:
            vetN = [0] * 20  # [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            vetC = [0] * 20  # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            if (len(peptideo) > posicao + 1):
                aminoN = peptideo[contador]
                aminoC = peptideo[len(peptideo) - contador - 2]
                # print(peptideo)
                # print(aminoN," ",aminoC)
                pN = self.buscarVetorG(self.amiacidos, aminoN)
                vetN[pN] = 1
                pC = self.buscarVetorG(self.amiacidos, aminoC)
                vetC[pC] = 1
            recursosN += vetN
            recursosC += vetC
            contador += 1
        return recursosN, recursosC

    # Ligação entre os grupos, frequência ou quantidade de ligação
    def ext_CKSAAGP(self, peptideo):
        peptideo = peptideo[0:len(peptideo)-1]
        grupos = ["0", "1", "2", "3", "4"]
        vet_lig_gp = self.preencher_lig_grups(grupos)
        vizinhos = self.gerarVizinhos(peptideo)
        recursos_CKSAAGP = [0] * 25
        qtd_CKSAAGP = [0] * 25
        for lig in vizinhos:
            lig_gp = str(self.buscar_grupo(lig[0])) + str(self.buscar_grupo(lig[1]))
            pos_lig_gp = self.buscarVetorG(vet_lig_gp, lig_gp)
            qtd_CKSAAGP[pos_lig_gp] += 1
            recursos_CKSAAGP[pos_lig_gp] = qtd_CKSAAGP[pos_lig_gp] / len(vizinhos)
        return qtd_CKSAAGP

    def preencher_lig_grups(self, vetor_grupos):
        vetor_lig_grupos = []
        for i in range(len(vetor_grupos)):
            for j in range(len(vetor_grupos)):
                vetor_lig_grupos.append(vetor_grupos[i] + vetor_grupos[j])
        return vetor_lig_grupos

    # Método responsável por receber uma sequência e calcular as combinações com base o valor de K, defino no inicio do algoritmo.
    # Até o momento só leva em consideração da esquerda para direita.
    def gerarVizinhos(self, peptide_sequence):
        # print(peptide_sequence)
        posicao = 0
        combinacoes = []
        # percorre todos os minoácitos, começa do 0 e vai até o penultimo, pois o ultimo não tem o próximo
        k_vizinho = 1
        while posicao < (len(peptide_sequence) - 1):
            # print("Aqui!",peptide_sequence[posicao])
            contadorK = 1
            while contadorK <= k_vizinho and (contadorK + posicao) < len(peptide_sequence):
                x = peptide_sequence[posicao] + str(peptide_sequence[contadorK + posicao])
                combinacoes.append(x)
                # print(x)
                contadorK += 1
            posicao += 1
        # print(combinacoes)
        return combinacoes

    # Aqui para baixo é apenas métodos auxiliares
    def buscar_grupo(self, amino):
        # Grupo GAVLMI
        if (amino == "G" or amino == "A" or amino == "V" or amino == "L" or amino == "M" or amino == "I"):
            return 0
        # Grupo 2 FYW
        if (amino == "F" or amino == "Y" or amino == "W"):
            return 1
        # Grupo 3 KRH
        if (amino == "K" or amino == "R" or amino == "H"):
            return 2
        # Grupo 4 DE
        if (amino == "D" or amino == "E"):
            return 3
        # Grupo 3 STCPNQ
        if (amino == "S" or amino == "T" or amino == "C" or amino == "P" or amino == "N" or amino == "Q"):
            return 4


if __name__ == '__main__':
    d = Descritores()
    vetorRecursos = d.ext_CKSAAGP("AACYCY")  # AA, AC,CY,YC,CY
    print(vetorRecursos)
