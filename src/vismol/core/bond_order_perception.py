"""
bond_order_perception.py

Percepcao de ordem de ligacao (single/double/triple) a partir de:
    - elementos (lista de simbolos, ex. ['C','C','O','H',...])
    - grafo de conectividade (lista de pares de indices ja determinada por
      raios covalentes)

Hipoteses (validas para XYZ organicos com H explicito):
    - todos os hidrogenios estao presentes;
    - molecula de camada fechada (sem radicais/valencias abertas);
    - sem metais de transicao (organico leve).

Metodo: variante do algoritmo de Wang & Case (Antechamber 'bondtype',
J. Mol. Graph. Model. 25 (2006) 247-260). Cada atomo recebe uma penalidade
(APS) em funcao da valencia total resultante da atribuicao de ordens. Busca-se
a combinacao de ordens que MINIMIZA a penalidade total (tps). Regras duras
resolvem os casos triviais antes da busca, e o backtracking com poda cobre o
resto.

API principal:
    perceive_bond_orders(elements, bonds) -> dict {(i,j): order}
        bonds: iteravel de pares (i,j). A chave de saida e sempre normalizada
        (min, max). order in {1,2,3}.

Este modulo nao depende de numpy; usa apenas a biblioteca padrao.
"""

from collections import defaultdict


# ---------------------------------------------------------------------------
# Parametros quimicos
# ---------------------------------------------------------------------------

# Valencia-alvo "normal" (numero de ligacoes covalentes que o atomo neutro
# de camada fechada costuma formar). Usada para o caso comum.
NORMAL_VALENCE = {
    'H': 1, 'D': 1,
    'B': 3,
    'C': 4, 'Si': 4,
    'N': 3, 'P': 3,
    'O': 2, 'S': 2,
    'F': 1, 'Cl': 1, 'Br': 1, 'I': 1,
}

# Penalty scores (APS) ao estilo Wang&Case, em DUAS camadas:
#
#   1) valencias quimicamente IMPOSSIVEIS (ex.: carbono com valencia 5) tem
#      custo PENALTY_FORBIDDEN (proibicao dura). A busca nunca as escolhe se
#      houver qualquer alternativa.
#   2) valencias PLAUSIVEIS recebem um custo pequeno: 0 para o estado neutro
#      ideal, e >0 para estados carregados comuns (N+ em nitro/amonio, O- em
#      carboxilato, S/P hipervalentes...). Assim o algoritmo so usa um estado
#      carregado quando a geometria/conectividade realmente exige, e prefere
#      a solucao com menos cargas formais (menor tps).
#
# Valencia_total = soma das ordens das ligacoes do atomo.

PENALTY_FORBIDDEN = 1000   # valencia impossivel: proibicao dura
PENALTY_BIG = 64           # default para valencias nao listadas

ATOM_PENALTY = {
    'H':  {1: 0},
    'D':  {1: 0},
    'B':  {3: 0, 4: 1},
    # Carbono: 4 e o unico estavel; 3 (carbanion/carbeno) custa alto; 5 proibido.
    'C':  {4: 0, 3: 4, 2: 8, 5: PENALTY_FORBIDDEN},
    'Si': {4: 0, 3: 4},
    # Nitrogenio: 3 neutro; 4 = N+ (amonio, nitro); 2 = amideto/N-.
    'N':  {3: 0, 4: 2, 2: 4, 5: PENALTY_FORBIDDEN},
    # Fosforo: 3 neutro; 5 hipervalente (fosfato); 4 intermediario.
    'P':  {3: 0, 5: 1, 4: 3, 6: PENALTY_FORBIDDEN},
    # Oxigenio: 2 neutro; 1 = O- (carboxilato, oxido, fenolato); 3 = O+.
    'O':  {2: 0, 1: 2, 3: 4},
    # Enxofre: 2 neutro; 1 = S-; 4 e 6 hipervalentes (sulfoxido, sulfona/sulfato).
    'S':  {2: 0, 1: 2, 3: 3, 4: 1, 6: 1},
    'F':  {1: 0},
    'Cl': {1: 0},
    'Br': {1: 0},
    'I':  {1: 0},
}

# Ordem maxima permitida por elemento (evita coisas absurdas).
MAX_ORDER = {
    'H': 1, 'D': 1, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1,
    'O': 2, 'S': 3, 'N': 3, 'P': 3, 'C': 3, 'Si': 3, 'B': 2,
}


def _penalty(element, valence):
    """ Penalidade APS de um atomo dado o elemento e a valencia total. """
    table = ATOM_PENALTY.get(element)
    if table is None:
        # elemento desconhecido: aceita a valencia observada sem penalizar
        # (melhor nao inventar ordens para atomos que nao entendemos).
        return 0
    return table.get(valence, PENALTY_BIG)


def _normalize(i, j):
    return (i, j) if i <= j else (j, i)


# ---------------------------------------------------------------------------
# Nucleo
# ---------------------------------------------------------------------------

class _Perceptor:
    def __init__(self, elements, bonds):
        self.elements = list(elements)
        self.n = len(self.elements)
        # adjacencia: atomo -> lista de vizinhos
        self.adj = defaultdict(list)
        self.bond_list = []
        seen = set()
        for (i, j) in bonds:
            key = _normalize(int(i), int(j))
            if key in seen or key[0] == key[1]:
                continue
            seen.add(key)
            self.bond_list.append(key)
            self.adj[key[0]].append(key[1])
            self.adj[key[1]].append(key[0])
        # ordem corrente de cada ligacao (inicialmente 1)
        self.order = {b: 1 for b in self.bond_list}
        # indice de ligacoes para busca
        self.bond_index = {b: k for k, b in enumerate(self.bond_list)}

    # --- utilidades ---------------------------------------------------------

    def _atom_valence(self, atom):
        """ Soma das ordens das ligacoes do atomo na atribuicao corrente. """
        v = 0
        for nb in self.adj[atom]:
            v += self.order[_normalize(atom, nb)]
        return v

    def _max_order_for(self, b):
        ei = self.elements[b[0]]
        ej = self.elements[b[1]]
        return min(MAX_ORDER.get(ei, 3), MAX_ORDER.get(ej, 3))

    # --- regras duras -------------------------------------------------------

    def _apply_acid_models(self, fixed):
        """ Modelo 'acido' (Wang&Case / Zhang&Hou): um atomo central X (C,N,P,S,
            Cl,Br,I) ligado a DOIS OU MAIS oxigenios/enxofres TERMINAIS (grau 1)
            forma um oxianion ressonante (carboxilato, nitro, sulfato, fosfato,
            ...). Nesses grupos a atribuicao correta e: UMA ligacao X=O dupla e
            as demais X-O simples (os O simples sao os formalmente carregados).

            Fixa esse padrao por regra dura, evitando que a busca caia no
            empate de penalidade que produziria 'tudo simples' (errado) ou
            'todas duplas' (hipervalencia falsa).

            Atua so quando ha >=2 oxigenios/enxofres terminais; o caso de 1 so
            (ex.: carbonila, acido com -OH) e deixado para as penalidades, que
            ja o resolvem certo.
        """
        TERMINALS = ('O', 'S')
        CENTERS = ('C', 'N', 'P', 'S', 'Cl', 'Br', 'I')
        for a in range(self.n):
            if self.elements[a] not in CENTERS:
                continue
            # oxigenios/enxofres terminais ligados a 'a'
            term = [nb for nb in self.adj[a]
                    if self.elements[nb] in TERMINALS and len(self.adj[nb]) == 1]
            if len(term) < 2:
                continue
            # Quantas duplas X=O o centro comporta? Escolhemos o numero de
            # duplas (entre 1 e n_term) que minimiza a penalidade do atomo
            # central, dado que as outras ligacoes terminais sao simples e as
            # ligacoes nao-terminais ja contam sua ordem corrente. Isso da:
            #   - C, N  -> 1 dupla  (carboxilato, nitro)
            #   - S      -> 2 duplas (sulfato/sulfona, valencia 6)
            #   - P      -> 1 dupla  (fosfato, valencia 5)
            # de forma automatica, sem hard-coding por elemento.
            el = self.elements[a]
            # valencia ja comprometida por ligacoes NAO-terminais de 'a'
            committed = 0
            for nb in self.adj[a]:
                if not (self.elements[nb] in TERMINALS and len(self.adj[nb]) == 1):
                    committed += self.order[_normalize(a, nb)]
            n_term = len(term)
            best_d, best_pen = 1, None
            for d in range(1, n_term + 1):
                # d duplas + (n_term-d) simples nos terminais
                val = committed + 2 * d + (n_term - d)
                pen = _penalty(el, val)
                # penaliza tambem cada O que fica simples (O-, valencia 1)
                pen += (n_term - d) * _penalty('O', 1)
                # e cada O duplo fica valencia 2 (penalidade 0)
                if best_pen is None or pen < best_pen:
                    best_pen, best_d = pen, d
            # aplica: best_d duplas, resto simples
            count_d = 0
            for nb in term:
                b = _normalize(a, nb)
                if b in fixed:
                    continue
                if count_d < best_d:
                    self.order[b] = 2
                    count_d += 1
                else:
                    self.order[b] = 1
                fixed.add(b)
        return fixed

    def _apply_hard_rules(self, prefixed=None):
        """ Fixa apenas ordens GENUINAMENTE forcadas, sem bloquear estados de
            valencia alternativos. So fixa a unica ligacao de um atomo terminal
            cujo elemento tem um unico estado de valencia possivel (H, F, Cl,
            Br, I). Atomos com estados alternativos (C,N,O,S,P) ficam livres
            para a busca decidir via penalidade. Isso evita o early-fixing que
            antes travava nitro/carboxilato numa solucao ruim.

            Retorna (free_bonds, fixed_bonds).
        """
        SINGLE_STATE = {'H', 'D', 'F', 'Cl', 'Br', 'I'}
        fixed = set(prefixed) if prefixed else set()
        for b in self.bond_list:
            if b in fixed:
                continue
            ei, ej = self.elements[b[0]], self.elements[b[1]]
            # Ligacao a um atomo de estado unico e sempre simples se esse atomo
            # so quer valencia 1 (H e halogenios terminais).
            if ei in SINGLE_STATE or ej in SINGLE_STATE:
                self.order[b] = 1
                fixed.add(b)
        free = [b for b in self.bond_list if b not in fixed]
        return free, fixed

    # --- custo --------------------------------------------------------------

    def _total_penalty(self):
        tps = 0
        for a in range(self.n):
            tps += _penalty(self.elements[a], self._atom_valence(a))
        return tps

    # --- busca com backtracking e poda -------------------------------------

    def solve(self):
        fixed_pre = self._apply_acid_models(set())
        free, fixed = self._apply_hard_rules(prefixed=fixed_pre)
        free.sort(key=lambda b: (len(self.adj[b[0]]) + len(self.adj[b[1]])))

        self.best_tps = [float('inf')]
        self.best_assignment = [dict(self.order)]
        max_orders = {b: self._max_order_for(b) for b in free}

        # Para cada atomo, quantas de suas ligacoes sao "livres" (decididas
        # durante a busca). Um atomo so tem penalidade DEFINITIVA quando todas
        # as suas ligacoes livres ja foram atribuidas.
        free_bonds_of_atom = defaultdict(int)
        for b in free:
            free_bonds_of_atom[b[0]] += 1
            free_bonds_of_atom[b[1]] += 1

        # ordem de atribuicao: marcamos, a cada passo, quantas ligacoes livres
        # de cada atomo ja foram decididas.
        decided_of_atom = defaultdict(int)

        free_atoms = set()
        for b in free:
            free_atoms.add(b[0]); free_atoms.add(b[1])

        fixed_penalty = 0
        for a in range(self.n):
            if a not in free_atoms:
                fixed_penalty += _penalty(self.elements[a], self._atom_valence(a))

        def lower_bound():
            """ Limite inferior VALIDO: soma a penalidade apenas dos atomos cujas
                ligacoes livres ja foram TODAS decididas (penalidade definitiva).
                Atomos ainda incompletos contribuem 0 (otimista), garantindo que
                nunca superestimamos -> poda segura. """
            s = fixed_penalty
            for a in free_atoms:
                if decided_of_atom[a] == free_bonds_of_atom[a]:
                    s += _penalty(self.elements[a], self._atom_valence(a))
            return s

        def backtrack(k):
            if k == len(free):
                tps = fixed_penalty + sum(
                    _penalty(self.elements[a], self._atom_valence(a))
                    for a in free_atoms)
                if tps < self.best_tps[0]:
                    self.best_tps[0] = tps
                    self.best_assignment[0] = dict(self.order)
                return
            b = free[k]
            for o in range(1, max_orders[b] + 1):
                self.order[b] = o
                va = self._atom_valence(b[0])
                vb = self._atom_valence(b[1])
                if va <= self._elem_cap(b[0]) and vb <= self._elem_cap(b[1]):
                    decided_of_atom[b[0]] += 1
                    decided_of_atom[b[1]] += 1
                    if lower_bound() < self.best_tps[0]:
                        backtrack(k + 1)
                    decided_of_atom[b[0]] -= 1
                    decided_of_atom[b[1]] -= 1
            self.order[b] = 1

        backtrack(0)
        self.order = self.best_assignment[0]
        return self.order, self.best_tps[0]

    def _elem_cap(self, atom):
        """ Maior valencia PLAUSIVEL do atomo: a maior chave da tabela de
            penalidade cujo custo nao e proibitivo. Usado para poda dura. """
        el = self.elements[atom]
        table = ATOM_PENALTY.get(el)
        if table:
            plausible = [v for v, pen in table.items() if pen < PENALTY_FORBIDDEN]
            if plausible:
                return max(plausible)
        return NORMAL_VALENCE.get(el, 4)


def perceive_bond_orders(elements, bonds):
    """ Funcao principal. Ver docstring do modulo.

        elements : lista de simbolos quimicos (1 por atomo, indexado por 0..N-1)
        bonds    : iteravel de pares (i,j) de indices de atomos ligados

        Retorna: dict {(i,j): order} com i<j e order in {1,2,3}.
    """
    p = _Perceptor(elements, bonds)
    order, tps = p.solve()
    return dict(order), tps


def perceive_for_vismol(elements, index_bonds):
    """ Ponte para o formato do EasyHybrid/VismolObject.

        elements    : lista de simbolos (1 por atomo)
        index_bonds : array ACHATADO de pares [i0,j0, i1,j1, ...] (como
                      self.index_bonds no VismolObject)

        Retorna bond_order_list: lista de inteiros, UMA ordem por ligacao, na
        MESMA ordem em que os pares aparecem em index_bonds -- pronta para
        atribuir a self.bond_order_list.
    """
    ib = list(index_bonds)
    pairs = [(int(ib[2 * k]), int(ib[2 * k + 1])) for k in range(len(ib) // 2)]
    order_map, _tps = perceive_bond_orders(elements, pairs)
    out = []
    for (i, j) in pairs:
        out.append(order_map.get(_normalize(i, j), 1))
    return out


# ---------------------------------------------------------------------------
# Teste rapido com moleculas conhecidas
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    def show(name, elements, bonds, expected=None):
        order, tps = perceive_bond_orders(elements, bonds)
        print("\n== %s == (tps=%d)" % (name, tps))
        for b in sorted(order):
            i, j = b
            print("  %2d(%s) - %2d(%s) : %d" % (
                i, elements[i], j, elements[j], order[b]))
        if expected is not None:
            got = {b: order[b] for b in order}
            ok = all(got.get(_normalize(*k)) == v for k, v in expected.items())
            print("  -> esperado:", "OK" if ok else "DIVERGENTE")

    # Etano CH3-CH3 : tudo simples
    show("etano",
         ['C','C','H','H','H','H','H','H'],
         [(0,1),(0,2),(0,3),(0,4),(1,5),(1,6),(1,7)])

    # Eteno CH2=CH2 : C-C dupla
    show("eteno",
         ['C','C','H','H','H','H'],
         [(0,1),(0,2),(0,3),(1,4),(1,5)],
         expected={(0,1):2})

    # Etino HC#CH : C-C tripla
    show("etino",
         ['C','C','H','H'],
         [(0,1),(0,2),(1,3)],
         expected={(0,1):3})

    # CO2 O=C=O : duas duplas
    show("CO2",
         ['C','O','O'],
         [(0,1),(0,2)],
         expected={(0,1):2,(0,2):2})

    # Formaldeido H2C=O
    show("formaldeido",
         ['C','O','H','H'],
         [(0,1),(0,2),(0,3)],
         expected={(0,1):2})

    # HCN : C#N
    show("HCN",
         ['H','C','N'],
         [(0,1),(1,2)],
         expected={(1,2):3})

    # Benzeno C6H6 : deve kekulizar em 3 duplas alternadas
    show("benzeno",
         ['C','C','C','C','C','C','H','H','H','H','H','H'],
         [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),
          (0,6),(1,7),(2,8),(3,9),(4,10),(5,11)])

    print("\n--- casos dificeis ---")

    # Acido acetico CH3-COOH : C=O (dupla) e C-O-H (simples)
    show("acido acetico",
         ['C','C','O','O','H','H','H','H'],
         [(0,1),(1,2),(1,3),(3,4),(0,5),(0,6),(0,7)],
         expected={(1,2):2,(1,3):1})

    # Carboxilato CH3-COO(-) : ressonancia, sem H no O. Antechamber atribui
    # uma dupla e uma simples (Kekule); ambos O ficam com valencia 2 e 1.
    show("acetato (anion)",
         ['C','C','O','O','H','H','H'],
         [(0,1),(1,2),(1,3),(0,4),(0,5),(0,6)])

    # Nitrometano CH3-NO2 : N(+) com uma dupla N=O e uma simples N-O(-)
    show("nitrometano",
         ['C','N','O','O','H','H','H'],
         [(0,1),(1,2),(1,3),(0,4),(0,5),(0,6)])

    # Acetonitrila CH3-C#N
    show("acetonitrila",
         ['C','C','N','H','H','H'],
         [(0,1),(1,2),(0,3),(0,4),(0,5)],
         expected={(1,2):3})

    # Butadieno CH2=CH-CH=CH2 : duas duplas terminais, simples no meio
    show("1,3-butadieno",
         ['C','C','C','C','H','H','H','H','H','H'],
         [(0,1),(1,2),(2,3),(0,4),(0,5),(1,6),(2,7),(3,8),(3,9)],
         expected={(0,1):2,(1,2):1,(2,3):2})

    # Piridina C5H5N : anel aromatico com N
    show("piridina",
         ['N','C','C','C','C','C','H','H','H','H','H'],
         [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),
          (1,6),(2,7),(3,8),(4,9),(5,10)])

    # Naftaleno C10H8 : dois aneis fundidos
    show("naftaleno",
         ['C','C','C','C','C','C','C','C','C','C','H','H','H','H','H','H','H','H'],
         [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),   # anel 1
          (4,6),(6,7),(7,8),(8,9),(9,5),         # anel 2 (compartilha 4-5)
          (0,10),(1,11),(2,12),(3,13),(6,14),(7,15),(8,16),(9,17)])

    print("\n--- oxianions e zwitterions ---")

    # Sulfato SO4(2-) : S central, 4 O terminais. S hipervalente, 2 duplas
    show("sulfato",
         ['S','O','O','O','O'],
         [(0,1),(0,2),(0,3),(0,4)])

    # Fosfato PO4(3-) : P central, 4 O terminais
    show("fosfato",
         ['P','O','O','O','O'],
         [(0,1),(0,2),(0,3),(0,4)])

    # Glicina zwitterion +H3N-CH2-COO- : amonio + carboxilato
    show("glicina zwitterion",
         ['N','C','C','O','O','H','H','H','H','H'],
         [(0,1),(1,2),(2,3),(2,4),    # N-C, C-C, C-O, C-O
          (0,5),(0,6),(0,7),          # N-H x3 (amonio)
          (1,8),(1,9)])               # C-H x2

    # Dimetil sulfoxido (CH3)2S=O : S com uma dupla
    show("DMSO",
         ['S','O','C','C','H','H','H','H','H','H'],
         [(0,1),(0,2),(0,3),
          (2,4),(2,5),(2,6),(3,7),(3,8),(3,9)])
