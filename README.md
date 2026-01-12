# Escoamento Monofásico Compressível em Meio Poroso (1D)
**Método dos Elementos Finitos com Firedrake**

Trabalho Final — **GA 033 – Elementos Finitos**

---

## Descrição

Este projeto resolve numericamente o escoamento monofásico compressível de um **gás ideal** em um **meio poroso unidimensional**, utilizando o **Método dos Elementos Finitos (MEF)** e a biblioteca **Firedrake**.

São considerados dois casos:
- **Problema transiente**
- **Problema estacionário**

Hipóteses:
- Meio poroso rígido (porosidade constante);
- Gás ideal (\( Z = 1 \));
- Ausência de termo fonte;
- Efeitos gravitacionais desprezados.

---

## Estrutura do Projeto
- darcy_1D_estacionario.py
- darcy_1D_transiente.py
- diego_ref_darcy_1d_transiente.py  (script de referecia do professor)
- test.py  (está em branco mesmo)

---

## Problema Transiente
Condições de contorno:
- \( p(0,t) = p_w \)

Condições iniciais: 
- \( p(x,0) = p_r \)

Discretização:
- Elementos Lagrange contínuos (CG), grau 1;
- Euler implícito no tempo;
- Problema não linear resolvido via Newton.

Saída:
compressible-flow-transiente.png


---

## Problema Estacionário
Condições de contorno:
- \( p(0,t) = p_w \)
- \( p(L,t) = p_r \)

Saída:
compressible-flow-steady-2dirichlet.png


---

## Dependências

- Python ≥ 3.9  
- Firedrake  
- NumPy  
- Matplotlib  

---

## Execução
Ative o ambiente virtual do Firedrake:

```bash
source ~/venv-firedrake/bin/activate

Em seguida, com o ambiente ativado, execute:
python compressible_flow_transient.py
python compressible_flow_steady.py
