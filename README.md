# Sieci neuronowe — od jednej do dwóch warstw

Implementacja prostych sieci neuronowych **od zera w Pythonie/NumPy** wraz z obszerną
prezentacją wyjaśniającą teorię i działanie kodu — od regresji logistycznej (sieć
jednowarstwowa) po uczenie wsteczną propagacją błędu (sieć dwuwarstwowa, problem XOR).

## Zawartość repozytorium

| Plik | Opis |
| --- | --- |
| `neural_networks_presentation_PL_EN.html` | Pełna **dwujęzyczna** prezentacja (PL/EN, przełącznik języka): teoria → obrazek → odniesienie do kodu (neuron, sigmoida, reguła delty, backpropagation, pełne kroki uczenia liczbowo). |
| `single_layer_network_in_python.ipynb` | Notebook (Google Colab) — sieć **jednowarstwowa** (klasyfikacja, reguła delty). |
| `two_layer_network_in_python.ipynb` | Notebook (Google Colab) — sieć **dwuwarstwowa** (warstwa ukryta, backpropagation, XOR). |

## Otwórz w Google Colab

- Sieć jednowarstwowa: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/finchWro/NeuralNetwork/blob/main/single_layer_network_in_python.ipynb)
- Sieć dwuwarstwowa: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/finchWro/NeuralNetwork/blob/main/two_layer_network_in_python.ipynb)

## Prezentacja

Otwórz plik `neural_networks_presentation_PL_EN.html` w przeglądarce. Prezentacja jest **dwujęzyczna** — w prawym górnym rogu znajduje się przełącznik `PL / EN`. Obejmuje m.in.:

- **Część 1 — sieć jednowarstwowa:** neuron biologiczny jako inspiracja, przejście od
  regresji liniowej do sigmoidy, architektura i inicjalizacja wag, forward pass, błąd
  i koszt (MSE), reguła delty oraz pełny krok uczenia rozpisany liczbowo.
- **Część 2 — sieć dwuwarstwowa:** dlaczego potrzebna jest warstwa ukryta (XOR), rola
  biasu, forward pass dla dwóch warstw, idea i pełne wzory backpropagation oraz
  kompletny krok uczenia rozpisany liczbowo.

## Wymagania

- Python 3 oraz `numpy` (rdzeń implementacji).
- `matplotlib` — wykresy (np. krzywa MSE).

```bash
pip install numpy matplotlib
```

## Autor

Błażej Zięba — 15.06.2026
