# MLforKids Image Classification Project

Este projeto utiliza a biblioteca TensorFlow para treinar um modelo de classificação de imagens, utilizando dados de treinamento fornecidos pelo site Machine Learning for Kids.

## Estrutura do Projeto

- `main.py`: Script principal que contém a classe `MLforKidsImageProject` para treinar e fazer previsões com o modelo de aprendizado de máquina.

## Como Usar

1. Clone o repositório:
    ```bash
    git clone https://github.com/bianca-bpas/mlforkids-IC.git
    ```

2. Instale as dependências necessárias:
    ```bash
    pip install tensorflow tensorflow-hub
    ```

3. Execute o script principal:
    ```bash
    python main.py
    ```

## Funcionalidades

- **Treinamento do Modelo**: Baixa imagens de treinamento do site Machine Learning for Kids e treina um modelo de classificação de imagens.
- **Previsão**: Faz previsões sobre novas imagens enviadas pelo usuário.
- **Teste**: Avalia a precisão do modelo utilizando um conjunto de imagens de teste.

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](./LICENSE) para mais detalhes.