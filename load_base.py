from descritores import Descritores


class Load_dados:
    def __init__(self):
        '''
        self.negativo_t = open('basedados/main/negativos_treinamento.txt', 'r')
        self.positivo_t = open('basedados/main/positivos_treinamento.txt', 'r')
        self.negativo_validacao = open('basedados/main/negativos_validacao_externa.txt', 'r')
        self.positivo_validacao = open('basedados/main/positivos_validacao_externa.txt', 'r')

        '''
        self.negativo_t = open('basedados/alternative/negativo_treinamento.txt','r')
        self.positivo_t = open('basedados/alternative/positivo_treinamento.txt','r')
        self.negativo_validacao = open('basedados/alternative/negativos_validacao_externa.txt','r')
        self.positivo_validacao = open('basedados/alternative/positivos_validacao_externa.txt','r')



    def load_aac(self):
        dados_treinamento = []
        classes_treinamento = []
        dados_validacao = []
        classes_validacao = []
        d = Descritores()
        for i in self.negativo_t:
            # print(len(i))
            recurso = d.composicao_aminoacidos(i)
            dados_treinamento.append(recurso)
            classes_treinamento.append(0)

        for i in self.positivo_t:
            recurso = d.composicao_aminoacidos(i)
            dados_treinamento.append(recurso)
            classes_treinamento.append(1)

        for i in self.negativo_validacao:
            recurso = d.composicao_aminoacidos(i)
            # print(len(recurso))
            dados_validacao.append(recurso)
            classes_validacao.append(0)

        for i in self.positivo_validacao:
            recurso = d.composicao_aminoacidos(i)
            # print(len(recurso))
            dados_validacao.append(recurso)
            classes_validacao.append(1)

        return dados_treinamento, classes_treinamento, dados_validacao, classes_validacao

    def load_dipeptideo(self):
        dados_treinamento = []
        classes_treinamento = []
        dados_validacao = []
        classes_validacao = []
        d = Descritores()
        for i in self.negativo_t:
            recurso = d.depeptideo(i)
            dados_treinamento.append(recurso)
            classes_treinamento.append(0)

        for i in self.positivo_t:
            recurso = d.depeptideo(i)
            dados_treinamento.append(recurso)
            classes_treinamento.append(1)

        for i in self.negativo_validacao:
            recurso = d.depeptideo(i)
            # print(len(recurso))
            dados_validacao.append(recurso)
            classes_validacao.append(0)

        for i in self.positivo_validacao:
            recurso = d.depeptideo(i)
            # print(len(recurso))
            dados_validacao.append(recurso)
            classes_validacao.append(1)

        return dados_treinamento, classes_treinamento, dados_validacao, classes_validacao

    def load_Nterminal_Cterminal(self):
        dados_treinamento = []
        classes_treinamento = []
        dados_validacao = []
        classes_validacao = []
        d = Descritores()
        totalTerminal = 9
        for i in self.negativo_t:
            # print(len(i))
            vetN, vetC = d.perfilBinario2(i,totalTerminal)
            #recurso = d.composicao_aminoacidos(i)
            dados_treinamento.append(vetN+vetC)
            classes_treinamento.append(0)

        for i in self.positivo_t:
            vetN, vetC = d.perfilBinario2(i,totalTerminal)
            dados_treinamento.append(vetN+vetC)
            classes_treinamento.append(1)

        for i in self.negativo_validacao:
            vetN, vetC = d.perfilBinario2(i,totalTerminal)
            dados_validacao.append(vetN+vetC)
            classes_validacao.append(0)

        for i in self.positivo_validacao:
            vetN, vetC = d.perfilBinario2(i,totalTerminal)
            dados_validacao.append(vetN+vetC)
            classes_validacao.append(1)
        return dados_treinamento, classes_treinamento, dados_validacao, classes_validacao

    def load_hibrido_AAC_NC(self):
        dados_treinamento = []
        classes_treinamento = []
        dados_validacao = []
        classes_validacao = []
        d = Descritores()
        totalTerminal = 9
        for i in self.negativo_t:
            # print(len(i))
            vetN, vetC = d.perfilBinario2(i, totalTerminal)
            recurso_aac = d.composicao_aminoacidos(i)
            dados_treinamento.append(recurso_aac + vetN + vetC)
            classes_treinamento.append(0)

        for i in self.positivo_t:
            vetN, vetC = d.perfilBinario2(i, totalTerminal)
            recurso_aac = d.composicao_aminoacidos(i)
            dados_treinamento.append(recurso_aac + vetN + vetC)
            classes_treinamento.append(1)

        for i in self.negativo_validacao:
            vetN, vetC = d.perfilBinario2(i, totalTerminal)
            recurso_aac = d.composicao_aminoacidos(i)
            dados_validacao.append(recurso_aac +vetN + vetC)
            classes_validacao.append(0)

        for i in self.positivo_validacao:
            vetN, vetC = d.perfilBinario2(i, totalTerminal)
            recurso_aac = d.composicao_aminoacidos(i)
            dados_validacao.append(recurso_aac +vetN + vetC)
            classes_validacao.append(1)
        return dados_treinamento, classes_treinamento, dados_validacao, classes_validacao

    def load_hibrido_AAC_DPC(self):
        dados_treinamento = []
        classes_treinamento = []
        dados_validacao = []
        classes_validacao = []
        d = Descritores()
        totalTerminal = 9
        for i in self.negativo_t:
            # print(len(i))
            vetN, vetC = d.perfilBinario2(i, totalTerminal)
            recurso_dpc = d.depeptideo(i)
            dados_treinamento.append(recurso_dpc + vetN + vetC)
            classes_treinamento.append(0)

        for i in self.positivo_t:
            vetN, vetC = d.perfilBinario2(i, totalTerminal)
            recurso_dpc = d.depeptideo(i)
            dados_treinamento.append(recurso_dpc + vetN + vetC)
            classes_treinamento.append(1)

        for i in self.negativo_validacao:
            vetN, vetC = d.perfilBinario2(i, totalTerminal)
            recurso_dpc = d.depeptideo(i)
            dados_validacao.append(recurso_dpc + vetN + vetC)
            classes_validacao.append(0)

        for i in self.positivo_validacao:
            vetN, vetC = d.perfilBinario2(i, totalTerminal)
            recurso_dpc = d.depeptideo(i)
            dados_validacao.append(recurso_dpc + vetN + vetC)
            classes_validacao.append(1)
        return dados_treinamento, classes_treinamento, dados_validacao, classes_validacao

    def load_CKSAAGP(self):
        dados_treinamento = []
        classes_treinamento = []
        dados_validacao = []
        classes_validacao = []
        d = Descritores()
        for i in self.negativo_t:
            # print(len(i))
            recurso = d.ext_CKSAAGP(i)+d.composicao_aminoacidos(i)
            dados_treinamento.append(recurso)
            classes_treinamento.append(0)

        for i in self.positivo_t:
            recurso = d.ext_CKSAAGP(i)+d.composicao_aminoacidos(i)
            dados_treinamento.append(recurso)
            classes_treinamento.append(1)

        for i in self.negativo_validacao:
            recurso = d.ext_CKSAAGP(i)+d.composicao_aminoacidos(i)
            # print(len(recurso))
            dados_validacao.append(recurso)
            classes_validacao.append(0)

        for i in self.positivo_validacao:
            recurso = d.ext_CKSAAGP(i)+d.composicao_aminoacidos(i)
            # print(len(recurso))
            dados_validacao.append(recurso)
            classes_validacao.append(1)

        return dados_treinamento, classes_treinamento, dados_validacao, classes_validacao


if __name__ == '__main__':
    l = Load_dados()
    x_trein, y_trein, x_val, y_val = l.load_aac()
    print(len(x_trein))
    print(len(y_trein))
