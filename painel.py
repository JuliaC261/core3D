#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Painel de controle do visualizador 3D.

Monta, no lado vazio da janela, quatro linhas espelhadas. Cada linha tem:

    [ display ]  > >  ( IN n )( OUT n )      (linhas 1 e 3)
    ( IN n )( OUT n )  < <  [ display ]      (linhas 2 e 4)

    - display: o retangulo cinza grande. Por enquanto nao faz nada, e so um
      lugar reservado; da para escrever nele com definir_texto_do_display().
    - LEDs: os dois triangulos. O de cima e o verde, o de baixo e o vermelho.
      Desligados, ficam cinza. Eles sempre apontam para os botoes.
    - botoes: os dois circulos. O da esquerda e o IN, o da direita e o OUT.

Comportamento padrao: apertar IN acende (ou apaga) o LED verde daquela linha,
apertar OUT faz o mesmo com o vermelho. Para plugar a logica de verdade, use o
gancho `ao_apertar`:

    painel.ao_apertar = lambda linha, tipo, ligado: print(linha, tipo, ligado)

As pecas de desenho 2D vem do modulo do menu, para nao ter duas copias do mesmo
codigo de vtkActor2D na pasta.
"""

from __future__ import annotations

import math

import vtk

from dropdown import (
    ALTURA_BARRA,
    Retangulo2D,
    _criar_ator_2d,
    _criar_texto,
)

# ----------------------------------------------------------------------------
# Medidas, em pixels
# ----------------------------------------------------------------------------
NUMERO_DE_LINHAS = 4

LARGURA_DISPLAY = 176
ALTURA_DISPLAY = 64
RAIO_BOTAO = 23
RAIO_FURO = 5
AFASTAMENTO_FUROS = 11               # distancia dos furos ate o centro
ESPACO_ENTRE_BOTOES = 18
LARGURA_LED = 22
ALTURA_LED = 24
AFASTAMENTO_LEDS = 15                # distancia de cada LED ate o meio da linha
ESPACO_HORIZONTAL = 16
ESPACO_ENTRE_LINHAS = 30
ALTURA_ROTULO = 16
FONTE_ROTULO = 13
LADOS_DO_CIRCULO = 44                # segmentos do poligono que finge ser circulo

# ----------------------------------------------------------------------------
# Cores
# ----------------------------------------------------------------------------
COR_DISPLAY = (0.85, 0.85, 0.85)
COR_CONTORNO_DISPLAY = (0.62, 0.62, 0.64)
COR_BOTAO = (0.89, 0.89, 0.90)
COR_CONTORNO_BOTAO = (0.74, 0.74, 0.76)
COR_FURO = (0.98, 0.98, 0.98)
COR_LED_APAGADO = (0.82, 0.82, 0.83)
COR_CONTORNO_LED = (0.70, 0.70, 0.72)
COR_LED_VERDE = (0.38, 0.72, 0.40)
COR_LED_VERMELHO = (0.72, 0.27, 0.27)
COR_ROTULO = (0.25, 0.25, 0.28)


# ----------------------------------------------------------------------------
# Forma generica (circulos e triangulos)
# ----------------------------------------------------------------------------
class Forma2D:
    """Poligono desenhado em pixels, com preenchimento e contorno."""

    def __init__(self, quantidade_de_pontos: int) -> None:
        self.quantidade = quantidade_de_pontos
        self.visivel = True
        self.tem_contorno = False

        self.pontos = vtk.vtkPoints()
        for _ in range(quantidade_de_pontos):
            self.pontos.InsertNextPoint(0.0, 0.0, 0.0)

        preenchimento = vtk.vtkPolyData()
        preenchimento.SetPoints(self.pontos)
        faces = vtk.vtkCellArray()
        faces.InsertNextCell(quantidade_de_pontos)
        for i in range(quantidade_de_pontos):
            faces.InsertCellPoint(i)
        preenchimento.SetPolys(faces)

        contorno = vtk.vtkPolyData()
        contorno.SetPoints(self.pontos)
        linhas = vtk.vtkCellArray()
        linhas.InsertNextCell(quantidade_de_pontos + 1)
        for i in range(quantidade_de_pontos):
            linhas.InsertCellPoint(i)
        linhas.InsertCellPoint(0)
        contorno.SetLines(linhas)

        self.ator = _criar_ator_2d(preenchimento)
        self.ator_contorno = _criar_ator_2d(contorno)
        self.ator_contorno.GetProperty().SetLineWidth(1.0)

    def definir_pontos(self, pontos: list) -> None:
        for indice, (x, y) in enumerate(pontos[: self.quantidade]):
            self.pontos.SetPoint(indice, x, y, 0.0)
        self.pontos.Modified()

    def pintar(self, cor: tuple, cor_do_contorno: tuple | None = None) -> None:
        self.ator.GetProperty().SetColor(*cor)
        self.tem_contorno = cor_do_contorno is not None
        if self.tem_contorno:
            self.ator_contorno.GetProperty().SetColor(*cor_do_contorno)
        self._aplicar_visibilidade()

    def mostrar(self, visivel: bool) -> None:
        self.visivel = bool(visivel)
        self._aplicar_visibilidade()

    def _aplicar_visibilidade(self) -> None:
        self.ator.SetVisibility(self.visivel)
        self.ator_contorno.SetVisibility(self.visivel and self.tem_contorno)

    def definir_camada(self, numero: int) -> None:
        self.ator.SetLayerNumber(numero)
        self.ator_contorno.SetLayerNumber(numero)

    def adicionar_em(self, renderizador: vtk.vtkRenderer) -> None:
        renderizador.AddViewProp(self.ator)
        renderizador.AddViewProp(self.ator_contorno)


def _circulo(centro: tuple, raio: float, lados: int = LADOS_DO_CIRCULO) -> list:
    """Pontos de um poligono regular, que de longe passa por circulo."""
    x, y = centro
    return [
        (
            x + raio * math.cos(2.0 * math.pi * i / lados),
            y + raio * math.sin(2.0 * math.pi * i / lados),
        )
        for i in range(lados)
    ]


def _triangulo(x: float, y: float, largura: float, altura: float,
               para_direita: bool) -> list:
    """Triangulo apontando para o lado, com (x, y) no centro da base."""
    ponta = x + largura if para_direita else x - largura
    return [(x, y + altura / 2.0), (x, y - altura / 2.0), (ponta, y)]


# ----------------------------------------------------------------------------
# Uma linha do painel
# ----------------------------------------------------------------------------
class LinhaDeControle:
    """Display + dois LEDs + dois botoes, tudo de uma linha so."""

    def __init__(self, numero: int, renderizador: vtk.vtkRenderer) -> None:
        self.numero = numero
        self.verde_ligado = False
        self.vermelho_ligado = False

        self.display = Retangulo2D()
        self.display.pintar(COR_DISPLAY, COR_CONTORNO_DISPLAY)
        self.display.definir_camada(0)
        self.display.adicionar_em(renderizador)

        self.texto_display = _criar_texto("", FONTE_ROTULO, COR_ROTULO)
        self.texto_display.GetTextProperty().SetJustificationToCentered()
        self.texto_display.SetLayerNumber(1)
        renderizador.AddViewProp(self.texto_display)

        self.led_verde = Forma2D(3)
        self.led_vermelho = Forma2D(3)
        for led in (self.led_verde, self.led_vermelho):
            led.definir_camada(0)
            led.adicionar_em(renderizador)

        self.botao_entrada = Forma2D(LADOS_DO_CIRCULO)
        self.botao_saida = Forma2D(LADOS_DO_CIRCULO)
        self.furos = []
        for botao in (self.botao_entrada, self.botao_saida):
            botao.pintar(COR_BOTAO, COR_CONTORNO_BOTAO)
            botao.definir_camada(0)
            botao.adicionar_em(renderizador)
            furos_do_botao = []
            for _ in range(4):
                furo = Forma2D(LADOS_DO_CIRCULO // 2)
                furo.pintar(COR_FURO, COR_CONTORNO_BOTAO)
                furo.definir_camada(1)
                furo.adicionar_em(renderizador)
                furos_do_botao.append(furo)
            self.furos.append(furos_do_botao)

        self.rotulo_entrada = _criar_texto(f"IN {numero}", FONTE_ROTULO, COR_ROTULO)
        self.rotulo_saida = _criar_texto(f"OUT {numero}", FONTE_ROTULO, COR_ROTULO)
        for rotulo in (self.rotulo_entrada, self.rotulo_saida):
            propriedade = rotulo.GetTextProperty()
            propriedade.SetFontFamilyToCourier()
            propriedade.SetJustificationToCentered()
            rotulo.SetLayerNumber(1)
            renderizador.AddViewProp(rotulo)

        self.centro_entrada = (0.0, 0.0)
        self.centro_saida = (0.0, 0.0)
        self.atualizar_leds()

    # -- posicionamento ------------------------------------------------------
    def dispor(self, x_inicial: float, centro_y: float, botoes_a_direita: bool) -> None:
        largura_botoes = 4 * RAIO_BOTAO + ESPACO_ENTRE_BOTOES

        if botoes_a_direita:
            x_display = x_inicial
            x_led = x_display + LARGURA_DISPLAY + ESPACO_HORIZONTAL
            x_botoes = x_led + LARGURA_LED + ESPACO_HORIZONTAL
        else:
            x_botoes = x_inicial
            x_led = x_botoes + largura_botoes + ESPACO_HORIZONTAL
            x_display = x_led + LARGURA_LED + ESPACO_HORIZONTAL

        self.display.definir_area(
            x_display, centro_y - ALTURA_DISPLAY / 2.0,
            x_display + LARGURA_DISPLAY, centro_y + ALTURA_DISPLAY / 2.0,
        )
        self.texto_display.SetDisplayPosition(
            int(x_display + LARGURA_DISPLAY / 2.0), int(centro_y)
        )

        # Os LEDs ficam colados no lado do display e apontam para os botoes.
        base_led = x_led + LARGURA_LED if not botoes_a_direita else x_led
        self.led_verde.definir_pontos(_triangulo(
            base_led, centro_y + AFASTAMENTO_LEDS, LARGURA_LED, ALTURA_LED,
            botoes_a_direita,
        ))
        self.led_vermelho.definir_pontos(_triangulo(
            base_led, centro_y - AFASTAMENTO_LEDS, LARGURA_LED, ALTURA_LED,
            botoes_a_direita,
        ))

        self.centro_entrada = (x_botoes + RAIO_BOTAO, centro_y)
        self.centro_saida = (
            x_botoes + 3 * RAIO_BOTAO + ESPACO_ENTRE_BOTOES, centro_y,
        )
        for botao, centro, furos in (
            (self.botao_entrada, self.centro_entrada, self.furos[0]),
            (self.botao_saida, self.centro_saida, self.furos[1]),
        ):
            botao.definir_pontos(_circulo(centro, RAIO_BOTAO))
            for indice, furo in enumerate(furos):
                angulo = math.pi / 4.0 + indice * math.pi / 2.0
                centro_furo = (
                    centro[0] + AFASTAMENTO_FUROS * math.cos(angulo),
                    centro[1] + AFASTAMENTO_FUROS * math.sin(angulo),
                )
                furo.definir_pontos(
                    _circulo(centro_furo, RAIO_FURO, LADOS_DO_CIRCULO // 2)
                )

        self.rotulo_entrada.SetDisplayPosition(
            int(self.centro_entrada[0]),
            int(centro_y + RAIO_BOTAO + ALTURA_ROTULO / 2.0 + 2),
        )
        self.rotulo_saida.SetDisplayPosition(
            int(self.centro_saida[0]),
            int(centro_y - RAIO_BOTAO - ALTURA_ROTULO / 2.0 - 2),
        )

    # -- estado --------------------------------------------------------------
    def atualizar_leds(self) -> None:
        self.led_verde.pintar(
            COR_LED_VERDE if self.verde_ligado else COR_LED_APAGADO,
            COR_CONTORNO_LED,
        )
        self.led_vermelho.pintar(
            COR_LED_VERMELHO if self.vermelho_ligado else COR_LED_APAGADO,
            COR_CONTORNO_LED,
        )

    def acertou(self, centro: tuple, x: float, y: float) -> bool:
        return math.hypot(x - centro[0], y - centro[1]) <= RAIO_BOTAO


# ----------------------------------------------------------------------------
# O painel inteiro
# ----------------------------------------------------------------------------
class PainelDeControle:
    """Quatro linhas de controle desenhadas no lado vazio da janela."""

    def __init__(self, janela: vtk.vtkRenderWindow, lado: str = "esquerda",
                 quantidade: int = NUMERO_DE_LINHAS) -> None:
        self.janela = janela
        self.lado = lado
        self.tamanho_desenhado = (0, 0)
        self.ao_apertar = None            # gancho: (numero, "in"/"out", ligado)

        janela.SetNumberOfLayers(max(2, janela.GetNumberOfLayers()))
        self.renderizador = vtk.vtkRenderer()
        self.renderizador.SetLayer(1)
        self.renderizador.InteractiveOff()
        if lado.lower().startswith("esq"):
            self.renderizador.SetViewport(0.0, 0.0, 0.5, 1.0)
        else:
            self.renderizador.SetViewport(0.5, 0.0, 1.0, 1.0)
        janela.AddRenderer(self.renderizador)

        self.linhas = [
            LinhaDeControle(numero + 1, self.renderizador)
            for numero in range(quantidade)
        ]

        self.dispor()
        janela.AddObserver("StartEvent", self._antes_de_renderizar)

    # -- posicionamento ------------------------------------------------------
    def _antes_de_renderizar(self, obj, evento) -> None:
        if self.janela.GetSize() != self.tamanho_desenhado:
            self.dispor()

    def dispor(self) -> None:
        largura, altura = self.janela.GetSize()
        self.tamanho_desenhado = (largura, altura)

        x0, _, x1, _ = self.renderizador.GetViewport()
        esquerda, direita = x0 * largura, x1 * largura

        largura_total = (
            LARGURA_DISPLAY + LARGURA_LED + 4 * RAIO_BOTAO
            + ESPACO_ENTRE_BOTOES + 2 * ESPACO_HORIZONTAL
        )
        inicio = esquerda + max(12.0, (direita - esquerda - largura_total) / 2.0)

        altura_da_linha = max(ALTURA_DISPLAY, 2 * RAIO_BOTAO + 2 * ALTURA_ROTULO)
        altura_total = (
            len(self.linhas) * altura_da_linha
            + (len(self.linhas) - 1) * ESPACO_ENTRE_LINHAS
        )
        topo_util = altura - ALTURA_BARRA
        primeiro_centro = (
            topo_util - max(20.0, (topo_util - altura_total) / 2.0)
            - altura_da_linha / 2.0
        )

        for indice, linha in enumerate(self.linhas):
            centro_y = primeiro_centro - indice * (
                altura_da_linha + ESPACO_ENTRE_LINHAS
            )
            # As linhas alternam de lado, como no desenho original.
            linha.dispor(inicio, centro_y, botoes_a_direita=(indice % 2 == 0))

    # -- estado --------------------------------------------------------------
    def definir_led(self, numero: int, cor: str, ligado: bool) -> None:
        """Liga ou desliga um LED. `cor` e "verde" ou "vermelho"."""
        linha = self.linhas[numero - 1]
        if cor.lower().startswith("verd"):
            linha.verde_ligado = bool(ligado)
        else:
            linha.vermelho_ligado = bool(ligado)
        linha.atualizar_leds()
        self.janela.Render()

    def definir_texto_do_display(self, numero: int, texto: str) -> None:
        """Escreve algo no display daquela linha (vazio por padrao)."""
        self.linhas[numero - 1].texto_display.SetInput(texto)
        self.janela.Render()

    def estado(self, numero: int) -> dict:
        linha = self.linhas[numero - 1]
        return {"verde": linha.verde_ligado, "vermelho": linha.vermelho_ligado}

    # -- cliques -------------------------------------------------------------
    def clique(self, x: float, y: float) -> bool:
        """Trata um clique. Devolve True se algum botao foi apertado."""
        for linha in self.linhas:
            if linha.acertou(linha.centro_entrada, x, y):
                return self._apertar(linha, "in")
            if linha.acertou(linha.centro_saida, x, y):
                return self._apertar(linha, "out")
        return False

    def _apertar(self, linha: LinhaDeControle, tipo: str) -> bool:
        if tipo == "in":
            linha.verde_ligado = not linha.verde_ligado
            ligado = linha.verde_ligado
        else:
            linha.vermelho_ligado = not linha.vermelho_ligado
            ligado = linha.vermelho_ligado

        linha.atualizar_leds()
        self.janela.Render()

        if callable(self.ao_apertar):
            self.ao_apertar(linha.numero, tipo, ligado)
        return True