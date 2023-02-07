{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/JRamola/NLP/blob/main/jyoti_sentiment_analysis_of_imdb_movie_reviews.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
        "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
        "trusted": true,
        "id": "7_EvoQIeyM9W"
      },
      "source": [
        "**Sentiment Analysis of IMDB Movie Reviews**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "X5XOjAwpyM9Y"
      },
      "source": [
        "**Problem Statement:**\n",
        "\n",
        "In this, we have to predict the number of positive and negative reviews based on sentiments by using different classification models."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "1424638f5259100af9f9a5c1b05bd23cf5b71e51",
        "id": "oQ2BhaeByM9Y"
      },
      "source": [
        "**Import necessary libraries**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0",
        "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a",
        "trusted": true,
        "id": "J3ppm-_HyM9Z"
      },
      "outputs": [],
      "source": [
        "#Load the libraries\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "import nltk\n",
        "from sklearn.feature_extraction.text import CountVectorizer\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.preprocessing import LabelBinarizer\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.stem.porter import PorterStemmer\n",
        "from wordcloud import WordCloud,STOPWORDS\n",
        "from nltk.stem import WordNetLemmatizer\n",
        "from nltk.tokenize import word_tokenize,sent_tokenize\n",
        "from bs4 import BeautifulSoup\n",
        "import spacy\n",
        "import re,string,unicodedata\n",
        "from nltk.tokenize.toktok import ToktokTokenizer\n",
        "from nltk.stem import LancasterStemmer,WordNetLemmatizer\n",
        "from sklearn.linear_model import LogisticRegression,SGDClassifier\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.svm import SVC\n",
        "from textblob import TextBlob\n",
        "from textblob import Word\n",
        "from sklearn.metrics import classification_report,confusion_matrix,accuracy_score\n",
        "\n",
        "import os\n",
        "#print(os.listdir(\"../input\"))\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "be1b642cce343f7a8f68f8c91f7c50372cdf4381",
        "id": "NxMXUHh8yM9b"
      },
      "source": [
        "**Import the training dataset**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "_uuid": "4c593c17588723c0b0b0f19851cb70a8447ced76",
        "scrolled": true,
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 380
        },
        "id": "j5E5xuIwyM9b",
        "outputId": "5f7a9603-e7ee-4dd8-de82-13bf2a9220d5"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(518, 2)\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                              review sentiment\n",
              "0  One of the other reviewers has mentioned that ...  positive\n",
              "1  A wonderful little production. <br /><br />The...  positive\n",
              "2  I thought this was a wonderful way to spend ti...  positive\n",
              "3  Basically there's a family where a little boy ...  negative\n",
              "4  Petter Mattei's \"Love in the Time of Money\" is...  positive\n",
              "5  Probably my all-time favorite movie, a story o...  positive\n",
              "6  I sure would like to see a resurrection of a u...  positive\n",
              "7  This show was an amazing, fresh & innovative i...  negative\n",
              "8  Encouraged by the positive comments about this...  negative\n",
              "9  If you like original gut wrenching laughter yo...  positive"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-590cd6a0-c8d9-419d-8811-f1861c54618a\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>review</th>\n",
              "      <th>sentiment</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>One of the other reviewers has mentioned that ...</td>\n",
              "      <td>positive</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>A wonderful little production. &lt;br /&gt;&lt;br /&gt;The...</td>\n",
              "      <td>positive</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>I thought this was a wonderful way to spend ti...</td>\n",
              "      <td>positive</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Basically there's a family where a little boy ...</td>\n",
              "      <td>negative</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Petter Mattei's \"Love in the Time of Money\" is...</td>\n",
              "      <td>positive</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>Probably my all-time favorite movie, a story o...</td>\n",
              "      <td>positive</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>I sure would like to see a resurrection of a u...</td>\n",
              "      <td>positive</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>This show was an amazing, fresh &amp; innovative i...</td>\n",
              "      <td>negative</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>Encouraged by the positive comments about this...</td>\n",
              "      <td>negative</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>If you like original gut wrenching laughter yo...</td>\n",
              "      <td>positive</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-590cd6a0-c8d9-419d-8811-f1861c54618a')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-590cd6a0-c8d9-419d-8811-f1861c54618a button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-590cd6a0-c8d9-419d-8811-f1861c54618a');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ],
      "source": [
        "#importing the training data\n",
        "imdb_data=pd.read_csv('/content/IMDB Dataset111.csv')\n",
        "print(imdb_data.shape)\n",
        "imdb_data.head(10)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "1ad3773974351ed9bdf389b2847d7475b36c2295",
        "id": "DgkVGdQyyM9c"
      },
      "source": [
        "**Exploratery data analysis**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "_uuid": "7f11c83b1320c8982b36889145f7f770563674a8",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 174
        },
        "id": "HzSBoRVByM9c",
        "outputId": "09f2f7dd-5884-46d3-d770-2cbc449ce24d"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                                   review sentiment\n",
              "count                                                 518       518\n",
              "unique                                                518         2\n",
              "top     One of the other reviewers has mentioned that ...  negative\n",
              "freq                                                    1       273"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-09e2859e-fec4-4997-8956-e6f2caa545ed\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>review</th>\n",
              "      <th>sentiment</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>518</td>\n",
              "      <td>518</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>unique</th>\n",
              "      <td>518</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>top</th>\n",
              "      <td>One of the other reviewers has mentioned that ...</td>\n",
              "      <td>negative</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>freq</th>\n",
              "      <td>1</td>\n",
              "      <td>273</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-09e2859e-fec4-4997-8956-e6f2caa545ed')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-09e2859e-fec4-4997-8956-e6f2caa545ed button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-09e2859e-fec4-4997-8956-e6f2caa545ed');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ],
      "source": [
        "#Summary of the dataset\n",
        "imdb_data.describe()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "453c3fd238f62ab8f649eb01771817e25bc0c77d",
        "id": "WpXs5aSPyM9c"
      },
      "source": [
        "**Sentiment count**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "_uuid": "cb6bb97b0f851947dcf341a1de5708a1f2bc64c1",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sH0-mtfhyM9c",
        "outputId": "2593b7db-987b-45c3-dd0f-e377cb34e354"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "negative    273\n",
              "positive    245\n",
              "Name: sentiment, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ],
      "source": [
        "#sentiment count\n",
        "imdb_data['sentiment'].value_counts()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "cHKjDd39yM9d"
      },
      "source": [
        "We can see that the dataset is balanced."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "f61964573faababe1f7897b77d32815a24954d2f",
        "id": "wp0PehOPyM9d"
      },
      "source": [
        "**Spliting the training dataset**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "_uuid": "d3aaabff555e07feb11c72cc3a6e457615975ffe",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yyq96wHyyM9d",
        "outputId": "d1a95fc9-6310-4c20-c536-bb689a2cdaf9"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(518,) (518,)\n",
            "(0,) (0,)\n"
          ]
        }
      ],
      "source": [
        "#split the dataset  \n",
        "#train dataset\n",
        "train_reviews=imdb_data.review[:40000]\n",
        "train_sentiments=imdb_data.sentiment[:40000]\n",
        "#test dataset\n",
        "test_reviews=imdb_data.review[40000:]\n",
        "test_sentiments=imdb_data.sentiment[40000:]\n",
        "print(train_reviews.shape,train_sentiments.shape)\n",
        "print(test_reviews.shape,test_sentiments.shape)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import nltk\n",
        "#nltk.download('omw-1.4')\n",
        "from nltk.corpus import wordnet\n",
        "\n",
        "nltk.download('wordnet')\n",
        "nltk.download('punkt')\n",
        "#nltk.download('averaged_perceptron_tagger')\n",
        "#nltk.download('maxent_ne_chunker')\n",
        "nltk.download('words')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KcOhlPNcJpic",
        "outputId": "9da0faa4-5688-4351-f8e0-5b20fd9c29e0"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package wordnet to /root/nltk_data...\n",
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Unzipping tokenizers/punkt.zip.\n",
            "[nltk_data] Downloading package words to /root/nltk_data...\n",
            "[nltk_data]   Unzipping corpora/words.zip.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "True"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "90da29c3b79f46f41d7391a2a116065b616d0fac",
        "id": "ipxTYsVOyM9e"
      },
      "source": [
        "**Text normalization**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "_uuid": "f000c43d91f68f6668539f089c6a54c5ce3bd819",
        "trusted": true,
        "id": "L-YhKh-MyM9e"
      },
      "outputs": [],
      "source": [
        "#Tokenization of text\n",
        "tokenizer=ToktokTokenizer()\n",
        "#Setting English stopwords\n",
        "stopword_list=nltk.corpus.stopwords.words('english')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "nltk.download('stopwords')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jvO5jRFaJz2Z",
        "outputId": "a7d243ec-22de-498c-b409-797415ec5e21"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Unzipping corpora/stopwords.zip.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "True"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "328b6e5977da3e055ad4b2e11a31e5e12ccf3b16",
        "id": "UqsY-oIOyM9e"
      },
      "source": [
        "**Removing html strips and noise text**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "_uuid": "6f6fcafbdadcdcb0c164e37d71fb9d1623f74d0a",
        "trusted": true,
        "id": "XeXFShrFyM9e"
      },
      "outputs": [],
      "source": [
        "#Removing the html strips\n",
        "def strip_html(text):\n",
        "    soup = BeautifulSoup(text, \"html.parser\")\n",
        "    return soup.get_text()\n",
        "\n",
        "#Removing the square brackets\n",
        "def remove_between_square_brackets(text):\n",
        "    return re.sub('\\[[^]]*\\]', '', text)\n",
        "\n",
        "#Removing the noisy text\n",
        "def denoise_text(text):\n",
        "    text = strip_html(text)\n",
        "    text = remove_between_square_brackets(text)\n",
        "    return text\n",
        "#Apply function on review column\n",
        "imdb_data['review']=imdb_data['review'].apply(denoise_text)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "88117b74761d1047924d6d70f76642faa0e706ac",
        "id": "Y6oD0U4lyM9f"
      },
      "source": [
        "**Removing special characters**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "_uuid": "219da72b025121fd98081df50ae0fcaace10cc9d",
        "trusted": true,
        "id": "r0H6aI9byM9f"
      },
      "outputs": [],
      "source": [
        "#Define function for removing special characters\n",
        "def remove_special_characters(text, remove_digits=True):\n",
        "    pattern=r'[^a-zA-z0-9\\s]'\n",
        "    text=re.sub(pattern,'',text)\n",
        "    return text\n",
        "#Apply function on review column\n",
        "imdb_data['review']=imdb_data['review'].apply(remove_special_characters)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "3b66eeabd5b7b8c251f8b8ddf331140a64bcd514",
        "id": "7Dz8HcviyM9f"
      },
      "source": [
        "**Text stemming\n",
        "**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "_uuid": "2295f2946e0ab74c220ad538d0e7adc04d23f697",
        "trusted": true,
        "id": "vrO67C4ayM9g"
      },
      "outputs": [],
      "source": [
        "#Stemming the text\n",
        "def simple_stemmer(text):\n",
        "    ps=nltk.porter.PorterStemmer()\n",
        "    text= ' '.join([ps.stem(word) for word in text.split()])\n",
        "    return text\n",
        "#Apply function on review column\n",
        "imdb_data['review']=imdb_data['review'].apply(simple_stemmer)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "e83107e4a281d84d7ae42b4e2c8d81b7ece438e4",
        "id": "O48Y505gyM9g"
      },
      "source": [
        "**Removing stopwords**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "metadata": {
        "_uuid": "5dbff82b4d2d188d8777b273a75d8ac714d38885",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ifXpY4KxyM9g",
        "outputId": "a856c8d0-c063-4d47-b78f-b7dbb652bab6"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'didn', 'won', \"wouldn't\", 'will', 'into', 'd', 'ours', 'we', 'a', 'more', 'y', \"don't\", 'her', 'whom', 'out', \"isn't\", \"couldn't\", 'shouldn', 'he', 'is', 'so', \"should've\", 'before', 'himself', 'until', 'and', 'needn', 'from', 'have', 'can', 'my', 'during', 'the', \"won't\", 'with', 'doesn', 'you', 'yours', 'because', \"weren't\", 'against', 'once', 'weren', 'it', 'off', 'myself', 'of', 'ain', \"needn't\", 'again', 've', 'not', \"you'll\", 'him', 'wasn', 'most', 'when', 'on', 'haven', 'being', 'itself', 'very', \"haven't\", 'isn', 'had', 'between', 'those', 'ourselves', 'mustn', 'were', \"shan't\", \"you've\", 'are', 't', 'any', 'ma', 'was', 'me', 'am', 'doing', 'down', 'been', 'now', 'hasn', 'both', 'that', 'or', \"mightn't\", 'them', 'at', 'themselves', 'by', 'herself', \"aren't\", \"shouldn't\", 'what', 'why', 'after', 'under', 'its', 'just', 'below', 'which', 'yourself', 'here', 'if', 'each', 'up', 'wouldn', 'your', 'be', 'but', 'aren', 'does', 'nor', 'there', 'in', \"didn't\", 'did', 'further', 'too', 'as', 'how', 'through', 'should', \"doesn't\", 'his', 'mightn', 'their', 'these', 'don', \"you'd\", 'they', 'having', 'where', 'do', \"you're\", 'for', 'm', 'll', 'this', 'some', 'same', \"that'll\", 'hadn', 'all', 'above', 'our', 'no', 's', 'hers', \"she's\", 'theirs', \"mustn't\", 'own', 'other', 'couldn', 'over', 'then', 'an', \"wasn't\", 'about', 'has', 'to', \"hadn't\", 'who', 'o', 'she', 'i', 'few', 'yourselves', 'such', 'shan', 're', 'only', 'than', \"hasn't\", \"it's\", 'while'}\n"
          ]
        }
      ],
      "source": [
        "#set stopwords to english\n",
        "stop=set(stopwords.words('english'))\n",
        "print(stop)\n",
        "\n",
        "#removing the stopwords\n",
        "def remove_stopwords(text, is_lower_case=False):\n",
        "    tokens = tokenizer.tokenize(text)\n",
        "    tokens = [token.strip() for token in tokens]\n",
        "    if is_lower_case:\n",
        "        filtered_tokens = [token for token in tokens if token not in stopword_list]\n",
        "    else:\n",
        "        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]\n",
        "    filtered_text = ' '.join(filtered_tokens)    \n",
        "    return filtered_text\n",
        "#Apply function on review column\n",
        "imdb_data['review']=imdb_data['review'].apply(remove_stopwords)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "b35e7499291173119ed42287deac6f0cd96516e1",
        "id": "cwSZPnvdyM9h"
      },
      "source": [
        "**Normalized train reviews**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 57,
      "metadata": {
        "_kg_hide-output": true,
        "_uuid": "b20c242bd091929ca896ea2c6e936ca00efe6ecf",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 191
        },
        "id": "6OPIoGkVyM9h",
        "outputId": "1b16b7ee-3e98-47b8-d819-ffe970c2a04a"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'one review ha mention watch 1 oz episod youll hook right thi exactli happen meth first thing struck oz wa brutal unflinch scene violenc set right word go trust thi show faint heart timid thi show pull punch regard drug sex violenc hardcor classic use wordit call oz nicknam given oswald maximum secur state penitentari focus mainli emerald citi experiment section prison cell glass front face inward privaci high agenda em citi home manyaryan muslim gangsta latino christian italian irish moreso scuffl death stare dodgi deal shadi agreement never far awayi would say main appeal show due fact goe show wouldnt dare forget pretti pictur paint mainstream audienc forget charm forget romanceoz doesnt mess around first episod ever saw struck nasti wa surreal couldnt say wa readi watch develop tast oz got accustom high level graphic violenc violenc injustic crook guard wholl sold nickel inmat wholl kill order get away well manner middl class inmat turn prison bitch due lack street skill prison experi watch oz may becom comfort uncomfort viewingthat get touch darker side'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 57
        }
      ],
      "source": [
        "#normalized train reviews\n",
        "norm_train_reviews=imdb_data.review[:400]\n",
        "norm_train_reviews[0]\n",
        "#convert dataframe to string\n",
        "#norm_train_string=norm_train_reviews.to_string()\n",
        "#Spelling correction using Textblob\n",
        "#norm_train_spelling=TextBlob(norm_train_string)\n",
        "#norm_train_spelling.correct()\n",
        "#Tokenization using Textblob\n",
        "#norm_train_words=norm_train_spelling.words\n",
        "#norm_train_words"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "d69462bb209a66cff86376dc8481d0c0140d894d",
        "id": "n06eYSH1yM9h"
      },
      "source": [
        "**Normalized test reviews**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 47,
      "metadata": {
        "_kg_hide-output": true,
        "_uuid": "c5d0d38bd9976150367e9d75f3b933774c96a1ab",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 122
        },
        "id": "kOloe3FLyM9h",
        "outputId": "aead9522-c96c-4af6-e6d3-647ef39222ff"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'empti lack lustr rendit classic novel wish peopl would stop mess classic clearli idea real intent point origin thi version differ felt ralph fienn version much wors though cast juliett brioch kathi ha got worst cast decis everanyway back thi version aim make stori relev contemporari set music style succe high art nit throwaway view raini day maybeth direct wa averag edit abysm wors old quinci deepak verma doe great turn hindley fact one britain wast talent part heath wa play great charm belief think cast strongest point thi project although talent director would made better use facil clear wa director hire didnt instil project passion deserv'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 47
        }
      ],
      "source": [
        "#Normalized test reviews\n",
        "norm_test_reviews=imdb_data.review[400:]\n",
        "norm_test_reviews[500]\n",
        "##convert dataframe to string\n",
        "#norm_test_string=norm_test_reviews.to_string()\n",
        "#spelling correction using Textblob\n",
        "#norm_test_spelling=TextBlob(norm_test_string)\n",
        "#print(norm_test_spelling.correct())\n",
        "#Tokenization using Textblob\n",
        "#norm_test_words=norm_test_spelling.words\n",
        "#norm_test_words"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "1c2a872ffcb6b8076fdbbba641af12081b6022ef",
        "id": "F3-wEFgYyM9i"
      },
      "source": [
        "**Bags of words model **\n",
        "\n",
        "It is used to convert text documents to numerical vectors or bag of words."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 48,
      "metadata": {
        "_uuid": "35cf9dcefb40b2dc520c5b0d559695324c46cc04",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-H_vTqQ2yM9i",
        "outputId": "4da9745e-dd0a-4627-fbe2-685d34f68e94"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "BOW_cv_train: (400, 98368)\n",
            "BOW_cv_test: (118, 98368)\n"
          ]
        }
      ],
      "source": [
        "#Count vectorizer for bag of words\n",
        "cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))\n",
        "#transformed train reviews\n",
        "cv_train_reviews=cv.fit_transform(norm_train_reviews)\n",
        "#transformed test reviews\n",
        "cv_test_reviews=cv.transform(norm_test_reviews)\n",
        "\n",
        "print('BOW_cv_train:',cv_train_reviews.shape)\n",
        "print('BOW_cv_test:',cv_test_reviews.shape)\n",
        "#vocab=cv.get_feature_names()-toget feature names"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "52371868f05ff9cf157280c5acf0f5bc71ee176d",
        "id": "LI3Xi0JhyM9i"
      },
      "source": [
        "**Term Frequency-Inverse Document Frequency model (TFIDF)**\n",
        "\n",
        "It is used to convert text documents to  matrix of  tfidf features."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 49,
      "metadata": {
        "_uuid": "afe6de957339921e05a6faeaf731f2272fd31946",
        "scrolled": false,
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NBnSHAXJyM9j",
        "outputId": "44c65f48-9bd9-4046-cbca-144c071e8412"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Tfidf_train: (400, 98368)\n",
            "Tfidf_test: (118, 98368)\n"
          ]
        }
      ],
      "source": [
        "#Tfidf vectorizer\n",
        "tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))\n",
        "#transformed train reviews\n",
        "tv_train_reviews=tv.fit_transform(norm_train_reviews)\n",
        "#transformed test reviews\n",
        "tv_test_reviews=tv.transform(norm_test_reviews)\n",
        "print('Tfidf_train:',tv_train_reviews.shape)\n",
        "print('Tfidf_test:',tv_test_reviews.shape)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "803e92b25faa738b10928a91de72d177d8dddf85",
        "id": "bLgQ8vHmyM9j"
      },
      "source": [
        "**Labeling the sentiment text**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 50,
      "metadata": {
        "_uuid": "60f5d496ce4109d1cdbf08f4284d4d26efd93922",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Cknvxa9fyM9j",
        "outputId": "357ec61b-4cb2-47a2-d624-3bfb7a6d174e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(518, 1)\n"
          ]
        }
      ],
      "source": [
        "#labeling the sentient data\n",
        "lb=LabelBinarizer()\n",
        "#transformed sentiment data\n",
        "sentiment_data=lb.fit_transform(imdb_data['sentiment'])\n",
        "print(sentiment_data.shape)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "21a80c94fb42e14391c627710c5d796c40aa7dde",
        "id": "dMoC0el0yM9j"
      },
      "source": [
        "**Split the sentiment tdata**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 58,
      "metadata": {
        "_kg_hide-output": true,
        "_uuid": "ca1e4cc917265ac98a72c37cffe57f27e9897408",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "s4Rm-kbcyM9j",
        "outputId": "63e180ec-0d51-4ee3-f263-43a6613a5389"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]]\n",
            "[[1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [0]\n",
            " [1]\n",
            " [0]\n",
            " [1]]\n"
          ]
        }
      ],
      "source": [
        "#Spliting the sentiment data\n",
        "train_sentiments=sentiment_data[:400]\n",
        "test_sentiments=sentiment_data[400:]\n",
        "print(train_sentiments)\n",
        "print(test_sentiments)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Qk-HkKHpyM9k"
      },
      "source": [
        "**Modelling the dataset**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "d5e45fdc9d062a5b9b9dd665ffe732776e196953",
        "id": "mcL0yy8byM9k"
      },
      "source": [
        "Let us build logistic regression model for both bag of words and tfidf features"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 59,
      "metadata": {
        "_uuid": "142d007421900550079a12ae8655bcae678ebaad",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "igpiTrsWyM9k",
        "outputId": "674e0246-e8d7-49d7-87ab-b64d54b37bd1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "LogisticRegression()\n",
            "LogisticRegression()\n"
          ]
        }
      ],
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "\n",
        "#training the model\n",
        "lr=LogisticRegression()\n",
        "#Fitting the model for Bag of words\n",
        "\n",
        "lr_bow=lr.fit(cv_train_reviews,train_sentiments)\n",
        "print(lr_bow)\n",
        "#Fitting the model for tfidf features\n",
        "lr_tfidf=lr.fit(tv_train_reviews,train_sentiments)\n",
        "print(lr_tfidf)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "07eb6d52eb32469e3be82e90af636d598a7b7c27",
        "id": "HUO_MVTayM9k"
      },
      "source": [
        "**Logistic regression model performane on test dataset**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 60,
      "metadata": {
        "_uuid": "52ad86935b76117f97b79e6672a3ba12352b9461",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qRpD-UA5yM9l",
        "outputId": "4673df87-748d-4dcf-e821-738ee58829b1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 1 0 1 1\n",
            " 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0\n",
            " 0 0 0 0 0 0 1 0 0 1 0 0 1 1 1 0 1 0 1 1 0 0 0 0 1 1 0 1 0 0 0 0 0 0 1 0 0\n",
            " 0 0 0 0 1 0 0]\n",
            "[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0\n",
            " 0 0 0 0 0 0 0]\n"
          ]
        }
      ],
      "source": [
        "#Predicting the model for bag of words\n",
        "\n",
        "lr_bow_predict=lr.predict(cv_test_reviews)\n",
        "print(lr_bow_predict)\n",
        "##Predicting the model for tfidf features\n",
        "lr_tfidf_predict=lr.predict(tv_test_reviews)\n",
        "print(lr_tfidf_predict)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CS-Kbi71yM9l"
      },
      "source": [
        "**Accuracy of the model**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 61,
      "metadata": {
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RJqUpNXTyM9l",
        "outputId": "e527f69f-a7be-4300-c61e-a505d68e65f1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "lr_bow_score : 0.5932203389830508\n",
            "lr_tfidf_score : 0.5338983050847458\n"
          ]
        }
      ],
      "source": [
        "#Accuracy score for bag of words\n",
        "lr_bow_score=accuracy_score(test_sentiments,lr_bow_predict)\n",
        "print(\"lr_bow_score :\",lr_bow_score)\n",
        "#Accuracy score for tfidf features\n",
        "lr_tfidf_score=accuracy_score(test_sentiments,lr_tfidf_predict)\n",
        "print(\"lr_tfidf_score :\",lr_tfidf_score)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "ac2ec8353acb5e0f548e1e4a590fbe6f34f4a686",
        "id": "gACfZPyHyM9l"
      },
      "source": [
        "**Print the classification report**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 62,
      "metadata": {
        "_uuid": "f89c7e7a6136d08790ffbf6bc4d0d05455f8555a",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_tm1qh7YyM9l",
        "outputId": "49db8a42-c590-4f50-e1d5-1ca303c08b76"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "    Positive       0.58      0.84      0.69        63\n",
            "    Negative       0.63      0.31      0.41        55\n",
            "\n",
            "    accuracy                           0.59       118\n",
            "   macro avg       0.61      0.58      0.55       118\n",
            "weighted avg       0.60      0.59      0.56       118\n",
            "\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "    Positive       0.53      0.98      0.69        63\n",
            "    Negative       0.50      0.02      0.04        55\n",
            "\n",
            "    accuracy                           0.53       118\n",
            "   macro avg       0.52      0.50      0.36       118\n",
            "weighted avg       0.52      0.53      0.39       118\n",
            "\n"
          ]
        }
      ],
      "source": [
        "#Classification report for bag of words \n",
        "lr_bow_report=classification_report(test_sentiments,lr_bow_predict,target_names=['Positive','Negative'])\n",
        "print(lr_bow_report)\n",
        "\n",
        "#Classification report for tfidf features\n",
        "lr_tfidf_report=classification_report(test_sentiments,lr_tfidf_predict,target_names=['Positive','Negative'])\n",
        "print(lr_tfidf_report)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "0d2e5ddcd69ff0fb52f05f17fc74a86e1b5e5b61",
        "id": "cZmATxU4yM9m"
      },
      "source": [
        "**Confusion matrix**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 63,
      "metadata": {
        "_uuid": "a36c058e834938559b7202f2142e61423a613b7a",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mzMh4ypIyM9m",
        "outputId": "02805561-7ae6-4062-9950-ac49b9779f19"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[17 38]\n",
            " [10 53]]\n",
            "[[ 1 54]\n",
            " [ 1 62]]\n"
          ]
        }
      ],
      "source": [
        "#confusion matrix for bag of words\n",
        "cm_bow=confusion_matrix(test_sentiments,lr_bow_predict,labels=[1,0])\n",
        "print(cm_bow)\n",
        "#confusion matrix for tfidf features\n",
        "cm_tfidf=confusion_matrix(test_sentiments,lr_tfidf_predict,labels=[1,0])\n",
        "print(cm_tfidf)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "8fde9753386e3593dc27c4e88e02bdc38462a018",
        "id": "NMJXveK1yM9m"
      },
      "source": [
        "**Stochastic gradient descent or Linear support vector machines for bag of words and tfidf features**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 64,
      "metadata": {
        "_uuid": "2211a9e97682195a0372b33e4da7267aad8548db",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "A2Dda72ryM9m",
        "outputId": "7c9f2b2a-f686-41d9-edc2-9a86a0f59b04"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "SGDClassifier(max_iter=500, random_state=42)\n",
            "SGDClassifier(max_iter=500, random_state=42)\n"
          ]
        }
      ],
      "source": [
        "#training the linear svm\n",
        "svm=SGDClassifier(loss='hinge',max_iter=500,random_state=42)\n",
        "#fitting the svm for bag of words\n",
        "svm_bow=svm.fit(cv_train_reviews,train_sentiments)\n",
        "print(svm_bow)\n",
        "#fitting the svm for tfidf features\n",
        "svm_tfidf=svm.fit(tv_train_reviews,train_sentiments)\n",
        "print(svm_tfidf)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "e9a7a973591c1d3cabaa1a47c57fa029d3752bab",
        "id": "wzeIjG0qyM9n"
      },
      "source": [
        "**Model performance on test data**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 65,
      "metadata": {
        "_uuid": "1a5ab738e04f0f9082c8d6ade6c2148cc398f8f3",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ji2nHXuZyM9n",
        "outputId": "e2ffefb2-fde4-4ff4-e688-2ff66253d402"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0\n",
            " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0\n",
            " 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0\n",
            " 0 0 0 0 1 0 0]\n",
            "[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            " 0 0 0 0 0 0 0]\n"
          ]
        }
      ],
      "source": [
        "#Predicting the model for bag of words\n",
        "svm_bow_predict=svm.predict(cv_test_reviews)\n",
        "print(svm_bow_predict)\n",
        "#Predicting the model for tfidf features\n",
        "svm_tfidf_predict=svm.predict(tv_test_reviews)\n",
        "print(svm_tfidf_predict)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "C-6GPuULyM9n"
      },
      "source": [
        "**Accuracy of the model**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 66,
      "metadata": {
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "REmB3RALyM9n",
        "outputId": "165b469f-4db1-4a58-b3af-28b29bf97832"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "svm_bow_score : 0.559322033898305\n",
            "svm_tfidf_score : 0.5338983050847458\n"
          ]
        }
      ],
      "source": [
        "#Accuracy score for bag of words\n",
        "svm_bow_score=accuracy_score(test_sentiments,svm_bow_predict)\n",
        "print(\"svm_bow_score :\",svm_bow_score)\n",
        "#Accuracy score for tfidf features\n",
        "svm_tfidf_score=accuracy_score(test_sentiments,svm_tfidf_predict)\n",
        "print(\"svm_tfidf_score :\",svm_tfidf_score)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "b1bd245f50902ad87ca28e48cbce64ec6a16ec5a",
        "id": "Z8BnM2EnyM9n"
      },
      "source": [
        "**Print the classification report**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 67,
      "metadata": {
        "_uuid": "d112bc5b4944330b567e19a7e04544a9a459f238",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qtLEwWrjyM9o",
        "outputId": "c816a8a6-e068-443e-f353-00469c4d74e5"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "    Positive       0.55      0.89      0.68        63\n",
            "    Negative       0.59      0.18      0.28        55\n",
            "\n",
            "    accuracy                           0.56       118\n",
            "   macro avg       0.57      0.54      0.48       118\n",
            "weighted avg       0.57      0.56      0.49       118\n",
            "\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "    Positive       0.53      1.00      0.70        63\n",
            "    Negative       0.00      0.00      0.00        55\n",
            "\n",
            "    accuracy                           0.53       118\n",
            "   macro avg       0.27      0.50      0.35       118\n",
            "weighted avg       0.29      0.53      0.37       118\n",
            "\n"
          ]
        }
      ],
      "source": [
        "#Classification report for bag of words \n",
        "svm_bow_report=classification_report(test_sentiments,svm_bow_predict,target_names=['Positive','Negative'])\n",
        "print(svm_bow_report)\n",
        "#Classification report for tfidf features\n",
        "svm_tfidf_report=classification_report(test_sentiments,svm_tfidf_predict,target_names=['Positive','Negative'])\n",
        "print(svm_tfidf_report)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "_uuid": "705fd8ae8bb5e6925852fffc906b6ffd769dbac0",
        "id": "dx6xrkUAyM9o"
      },
      "source": [
        "**Plot the confusion matrix**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 68,
      "metadata": {
        "_uuid": "49cde912705acbaef90d7a269cd27ea8a2815f03",
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "y9eiyZizyM9o",
        "outputId": "206f8546-9407-4e89-9513-66eb2265d9f5"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[10 45]\n",
            " [ 7 56]]\n",
            "[[ 0 55]\n",
            " [ 0 63]]\n"
          ]
        }
      ],
      "source": [
        "#confusion matrix for bag of words\n",
        "cm_bow=confusion_matrix(test_sentiments,svm_bow_predict,labels=[1,0])\n",
        "print(cm_bow)\n",
        "#confusion matrix for tfidf features\n",
        "cm_tfidf=confusion_matrix(test_sentiments,svm_tfidf_predict,labels=[1,0])\n",
        "print(cm_tfidf)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gaMhZx1gyM9p"
      },
      "source": [
        "**Multinomial Naive Bayes for bag of words and tfidf features**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 69,
      "metadata": {
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "y3uu6bO9yM9p",
        "outputId": "a1a40422-b465-48c6-bf7d-9975b99416c9"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "MultinomialNB()\n",
            "MultinomialNB()\n"
          ]
        }
      ],
      "source": [
        "#training the model\n",
        "mnb=MultinomialNB()\n",
        "#fitting the svm for bag of words\n",
        "mnb_bow=mnb.fit(cv_train_reviews,train_sentiments)\n",
        "print(mnb_bow)\n",
        "#fitting the svm for tfidf features\n",
        "mnb_tfidf=mnb.fit(tv_train_reviews,train_sentiments)\n",
        "print(mnb_tfidf)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "LaIsacv-yM9p"
      },
      "source": [
        "**Model performance on test data**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 70,
      "metadata": {
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DY4TPKK1yM9p",
        "outputId": "8aadf971-0a76-4bfc-dd70-c4bd9921b079"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0 0 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 1 1 0 0 0 0 0 1 0 1 0 1 1\n",
            " 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1 0 0 1 1 0 1 0 0 1 0 0 1 0 1 0 1 0 1 0\n",
            " 1 0 0 0 0 0 1 0 0 1 1 0 1 1 1 0 1 0 1 1 1 0 1 0 1 1 0 1 0 0 0 0 1 0 1 0 0\n",
            " 0 1 1 0 1 0 0]\n",
            "[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0\n",
            " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0\n",
            " 0 0 0 0 0 0 1 0 0 1 0 0 1 0 1 0 0 0 1 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1 0 0\n",
            " 0 0 0 0 1 0 0]\n"
          ]
        }
      ],
      "source": [
        "#Predicting the model for bag of words\n",
        "mnb_bow_predict=mnb.predict(cv_test_reviews)\n",
        "print(mnb_bow_predict)\n",
        "#Predicting the model for tfidf features\n",
        "mnb_tfidf_predict=mnb.predict(tv_test_reviews)\n",
        "print(mnb_tfidf_predict)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WCfF9sd-yM9q"
      },
      "source": [
        "**Accuracy of the model**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 71,
      "metadata": {
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XJNJKuZVyM9q",
        "outputId": "d37207b1-6d2e-4e88-ad73-ca3c43c2773c"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "mnb_bow_score : 0.6016949152542372\n",
            "mnb_tfidf_score : 0.5423728813559322\n"
          ]
        }
      ],
      "source": [
        "#Accuracy score for bag of words\n",
        "mnb_bow_score=accuracy_score(test_sentiments,mnb_bow_predict)\n",
        "print(\"mnb_bow_score :\",mnb_bow_score)\n",
        "#Accuracy score for tfidf features\n",
        "mnb_tfidf_score=accuracy_score(test_sentiments,mnb_tfidf_predict)\n",
        "print(\"mnb_tfidf_score :\",mnb_tfidf_score)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "p3BkVlrKyM9q"
      },
      "source": [
        "**Print the classification report**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 72,
      "metadata": {
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Tz5lnOYuyM9q",
        "outputId": "6ff6f743-ff22-4f47-eb67-2aed018b454b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "    Positive       0.61      0.71      0.66        63\n",
            "    Negative       0.59      0.47      0.53        55\n",
            "\n",
            "    accuracy                           0.60       118\n",
            "   macro avg       0.60      0.59      0.59       118\n",
            "weighted avg       0.60      0.60      0.60       118\n",
            "\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "    Positive       0.54      0.87      0.67        63\n",
            "    Negative       0.53      0.16      0.25        55\n",
            "\n",
            "    accuracy                           0.54       118\n",
            "   macro avg       0.54      0.52      0.46       118\n",
            "weighted avg       0.54      0.54      0.47       118\n",
            "\n"
          ]
        }
      ],
      "source": [
        "#Classification report for bag of words \n",
        "mnb_bow_report=classification_report(test_sentiments,mnb_bow_predict,target_names=['Positive','Negative'])\n",
        "print(mnb_bow_report)\n",
        "#Classification report for tfidf features\n",
        "mnb_tfidf_report=classification_report(test_sentiments,mnb_tfidf_predict,target_names=['Positive','Negative'])\n",
        "print(mnb_tfidf_report)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "iZJTgD6hyM9r"
      },
      "source": [
        "**Plot the confusion matrix**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 73,
      "metadata": {
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-C6qhZiPyM9r",
        "outputId": "835459cc-b251-4e01-884a-344dadd7a677"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[26 29]\n",
            " [18 45]]\n",
            "[[ 9 46]\n",
            " [ 8 55]]\n"
          ]
        }
      ],
      "source": [
        "#confusion matrix for bag of words\n",
        "cm_bow=confusion_matrix(test_sentiments,mnb_bow_predict,labels=[1,0])\n",
        "print(cm_bow)\n",
        "#confusion matrix for tfidf features\n",
        "cm_tfidf=confusion_matrix(test_sentiments,mnb_tfidf_predict,labels=[1,0])\n",
        "print(cm_tfidf)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GqjTe2qryM9r"
      },
      "source": [
        "**Let us see positive and negative words by using WordCloud.**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WBN5Lj-kyM9s"
      },
      "source": [
        "**Word cloud for positive review words**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 74,
      "metadata": {
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 348
        },
        "id": "XOdN9uB4yM9s",
        "outputId": "737ef874-becc-43ef-8491-545764bcfb81"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<function matplotlib.pyplot.show(*args, **kw)>"
            ]
          },
          "metadata": {},
          "execution_count": 74
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x720 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAlYAAAE6CAYAAADUexyjAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOy9d5Qd133n+bmVXw6dG92NHEgCYAKDJGYxSCQlUbJkW7ZX9jjI9vHs2J4da7QTPJ4z493j9czIO7ZnZVuOkqxg0VSgKImSSDEHkAABEEQOnePrfvm9inf/qEY3mh3QALoJkHyfcxoH/V71rfvqVd361i8KKSUNGjRo0KBBgwYNLh7lUk+gQYMGDRo0aNDgnUJDWDVo0KBBgwYNGqwQDWHVoEGDBg0aNGiwQjSEVYMGDRo0aNCgwQrREFYNGjRo0KBBgwYrRENYNWjQoEGDBg0arBCrIqyEEB8QQhwRQhwXQnx2NfbRoEGDBg0aNGhwuSFWuo6VEEIFjgL3AAPAbuCTUso3VnRHDRo0aNCgQYMGlxmrYbG6ETgupTwppXSArwIfWYX9NGjQoEGDBg0aXFashrBaA/Sf9fvA9GsNGjRo0KBBgwbvaLRLtWMhxKeBT0//ev2lmkeDBg0aNGjQoMF5MiGlbFnojdWwWA0C3Wf93jX92hyklH8ppdwlpdy1CnNo0KBBgwaXGU0brqd1y3su9TTefQhB8tZbyHzofrRM5lLP5p1C72JvrIbFajewWQixnlBQ/Szwc6uwnwtCt+J077wfK968qvsJfIfTe75FvTS+qvu5XFFUAyOaQtUthBAEvodbL+PWS0Cj8XeDdwct63fRvO56wpye1WNq6A2GD/9kVfexEuhWHFW3LvU03nWoyQSJ992MEolQ2XcApqYu9ZTe0ay4sJJSekKIfwn8AFCBv5FSHlzp/VwoQtGJZdYQy6xu2Jfn1lE1Y1X3cTmiqDqp9i20briJaKYTzYghFIXAc7ArU+SH3mDs5MvYlclLPdXLGj0TRUtY1IcLSNef+6YiMFsS+HUXr1B7y+cmNAWzNYk7VcWvOW/5/t9OmNEMieb1KOrqRl3USmOrOv7FoKg6RiyNDHyEos48VwlFxYimEaqKV6/g2RWEUNAjCQLfQzNjyMDDqeSRMgBAs+JoZozAc3CqBZh+vcHS6M3NaE1Zgupbv168G1mVq11K+Rjw2GqM3eDyRSgqzet2seaquzEiSYQQM+8pRgTNsIgkW4mk2ul77VHq5YkV2a+hxcnEe4hFWlAVDderU6mPk68M4HrVs2eIZSTJJtYTNTP4gUepNkK+3Ifn23PGVIRKMtZJOtaNplo4XoXJ0mkqtTHkalvcFEHbB3bScveVHP6Db1LrnytCjeY4Wz77AMWDg/R/8XkC21vd+byJ+OY2Nv72vQx/ay+j39/fMEA2WBRFM2jbdiuRdDterYSVaqWSGwCh0LLpJmLNPUjfRQLjR1/ErZfp2fURnEoeoajokQS5U3uZGnidWFM3zRtuACSqZpIffIOpvgMzoqvBIgiBubYboetAQ1i9FVyy4PUG7zyi6Q46tt4yT1TNIlBUjXTHVmqFEQYO/hAZ+Atst3xMPcHmNXeTjnVRd0tI6aNrERShcXzoScbyh2a2TUY72dR5B5aRwnZLqIrOmqZrGCsc4eTwUzPiShEqa5qvo7vlBvzAwfNtDC1GZ9M1HB/8MRPFE6y2mlBMDS1ugjL/OAohEJq6yDF+CxACoSlwiXbfYJURAqFrcOb8khLpenABNQ+j6Q4SLevo3/sYnl2l+9r7EYqClWgm3XUl/Xsexa2VaFp/LS2bbmTk0FPoVoL84GHyAwdJtm8m07OD0tgJmtZdR704ylT/68SyXTRvvJHS2CncWnHxCSgKQtNABkjXQ2gaRk83eksz0naw+/rwJqfdYqqK0dGO0dmB9H2c4RHc0THwl1ijVBUlEkFvyqKmUyhWBARI18UvlfHGJ/CKBfDPQ/wpCmo8jt7WippMIgw9/A4cB79Yws3lCMoVpHeO70RRQFFQDANr48ZwvRCg6DrCWNibIl13Wd+zMA30lhb05iaUSAQpJUG5jDM6hj+VD+d2zkFAaDoIEW4fTB8jVUHLZjHa2lBiURAKsl7DnZzCGx8nqNXPPfYlpiGsGqwYmc4rMWNN57zhK6pOZs1VK+ISTMe7ySbW0Tv6AsNTryNlgKaaWHqKSn3WPWJocTZ23IahRznY+21q9iSKotOR3cHatvdQqo4wPLk//Bzxtaxvv4XhyQMMjL+CFziYepytXfeysfNOKvUcNefSxSjY4yWO/l+P4lXtt9xaBVA+NsqhP/gmbq7SsFa9A9Hbm8j83P2o6QQAQbnK5D9+D7d/5LzHMmJpPLdOvTiGDHxqxdEZ12Dge9Ty4ZiV3CCJtk0omonn1Kjk+vDsCnZlElUzUPUI0WwnVqKJaLZrxrWqqPqS+7c2bSTzgXvxJieZ/M5jxK7ZSer9d6DG46F4Ghgk941HcEbHSNxwPam77kBrykIQ4E5OUfjxk5Rf2TNfXKkqZvcaojt3YG3cgJZOo0SsaasQSN9H2jZ+sUTlwOuUnnsBv7CEADwzbCZNfNf1xK7egZZJo5gWqMrsmHUbv1zGGRyivGcvtUNHZgXJNIplYW5Yj9HWit7eht7agtEVhr6osThNP/NxpLOwCz/3T/+M3du3+AQVBXPdWlK334K5tgclGgtFOCBtG69QpHboCKUXXgxF6RJo6TTZjz2ElklT+PGTVF7bjxKLknjPzcR3XYeWTs0IQOl5BNUqtUNHmPzmdwjql7e4agirBiuCohnEMl1hDMUyMGIZrHjTRQsrP3ABsIwUilCpuxVcr0rNnit80vFuEtEOTg4/TaEyMPP6UO41OrI7aMtcycjU6yhCpTVzBX7g0D++m7pTAMD1qgxM7GH7uodIxddQm1w5YSU0BaMlgZ6I4NsuzkRpwadGLWlhdaQR01YsKSVesb7gtkJXMZrjaHELoQj8uoszXsKvzl1QlYhOpDNDfShP4PmYzQm0hEXg+ji5Ml6pNiOezNYkejY2Y8jwK3YjxuodiDB0jK42tJYwe8zPl1DMC4sX9T0HRdFQVB1fBqh6BGSA79ooioqiGQSeg2ZGkIGH9D2kDJBnhMz0uScDD7dWIj9wkOLI8ekXJZ5dWXL/ajSC0dWJ1tJMrK+f1N13ARDYNkokgrVhPekP3EPl1b2k77sHxTLxy2XUeByjrZX0vXfjDI/g9PXPHTcWo+mnPorZ0z09FRmKnmIRJAjLRInFQstTextGZyeT//zNWevYAugtLTR94qNYmzaG1ibPx69Wkb6PUBQUy0SJx1AToTUrqNWoHT46bxxjTSet/+JTCF2b95ArNBWjrXXROQjLXPw9XSd+4y7S992NlkpNf+Y6fqEKikCJRtHbWtFbW7C2bCL3jX/GPnl6yfGMjnb0lmb0jg60gUEyH7yP2DU7QVGQrktQryM0DaHraKkUgePMnhuXMe9CYSUJPIfAdxGKihCXdx/qZFLQ2qwyNu5TLF2+5gFVM9HM2LLdU0IoGNH0Re83X+5jMPcaHZntZBPrmCydZrxwhGJ1eE7cVMxqQldN2rNXkUn0zJ2HHicIPBShoSgacasFXY2ypetepJy9iA0tjiI0TC1O6Ae7+O9DjVt0fPgaWu66AsXS8SsOpcPDCHX+eRnf3Eb3p96HnopiNMcZ/d4Ber/w1DyrlZ6N0f0L7yW5fQ2qpYfuRAnl46P0/fUz1AZmxWx0bTNb/92D9P7tM0TXNpO5aQN6MgKKYPhbexl6+JWZ4PnsezfReu92tISFkY1x6vNPMvLoaw2r1RIEgReuNUKAUC6d+/YSUZ0cJFjn0rH9LtxqkVh2DeVcH7X8MHY5R+eOe3BrBWJNPUz1v07guwuO49lVpvpfJ73mClQjAoDv1MIYq2W42dR4jOSdt1F5bR/l3a+iJuJk7rsHc20P1qZNGB3tuLkchSefws8XSLzvPSRuugEtlSK6bes8YeWXStSPn0CJxbB7+6ifOIk7Nha66KRETSSIXrmN+I27UGMxoldsxd51HYUfP7mwW1AIknfcGooqoH7sBKXnX8SdmEC6LkJVUWIxjLZWrE0b0TLpMLtvAZHh5nJMfee7M65coWskb30fWjqNX6tRfvFlvKn8gsfJHVski10Ioju3k/7AvaiJOF4+T/mlV6gdP45friAUBb2lmfiNNxDZvBGjs4PsRz7E2N99CX8ZGYhGRzvp++4metWV2H0DVF7bhzs6Fn52XUdvacbauIHakWOhu/Iy510nrDy7wuAbT2DG0qiahWpYqHr4o+kWqh6ZeU1RNISiIISCEKEIE4ryli6Qd91m8ZnfSfJ///ci3/ne5Rt4GB6b5aeUCwRCufjTz/NtTo88S654gpbUFrKJ9bSktjBZOsWpkWepOeECoggVhCAIvDnBrlIGTBSOUbPznOmbKRQVSUAQeJytGmy3yPDkfir13EXP+wzNd2yl/UPXUNjbx/iTYTxY9qaNZG5cP2/bwv4BKv/pm0R6mtj8mQ8ueg4GjodXrDH62P7pwHdJYnsXHR++lrYHdtL7108jvfAYCAFqzKT9wWuwR4sMP/IqbrGO2ZqgenoC6c0u3KPf30/umSMkd3az4bfuno3BabAoU0OHcOulcF3RLTTDmv3/9LqjGhFU1Zh90JtZc5Tp196+x9mtFRna/ziJtk34bp3hg08Q+B6B5zD0+o9Jtm9G1UwmTrxMafw0iqKSO7UHd9oS5VTzTJzaQxB4TPW/jlstYKXaQPrUCmMEy43RFAJpOxSffCq0GikKejaLsWYNSsSCwKf4nceoHTwEUlJ64SVi269CSYQWJ4SYaxmWkuLTz1Heuw93bBxp23Ped4dHsPv6CByH9D3vB1XF2riR0vMvEZTL86anxGOY69aCouBN5Jh69LEFXXL14yco734VJRHHzxcW/Kh+vkDx2ednx7YsYtdeg5ZOI22Hyt592P0DC/7tm92KZ9Cam0jedgtqIk5QrTL5nceo7tsfxt5N4wwMYvcPkn3oQaLbr8Ls7iK+6zoKT/xk6Tg1ILJ5IzIIqLy2j/wPfxx+R2fNpXZEofTyK8uL3boMeNcJq8B3yQ/PBjQjFBRFnVnUxPT/FUVD0Qw0I4JqRNCMKJoeIdW+mVT71rdsvqYpyGZUTPPyXlyl9JHB8k96iSTw7HNvuAz8wCVf7qNYGWTQ2ENHdiddLddjuxVODD8JgO1V8H2H4cn9jOWPLDwf6aJIFcctowiV40NP4HrzxWwgfZZtplnopji9AKtxk6b3bcadqjDwjy/OWJJq/ZPENrZgdc616EnXx81XUUxtfgmGs49H2WbgKy8SON7MNEuHh0lf3UNiWwdCVWaElZSgGBrSD+j/8gvUB896unyTUS6oezh1D3u8tCwrwdKIhYPfV7gp/KWmVhihVpiNTTqzvghFRREqqmaSbdqC7ZZw3PKM0NKMKGYsQ8v6G9DN2CX8BBdPvTRBvTQ/A9itlcid2jPnNd/3mOrbf9Y2xZnfpe9SGjtJaezkBc3DHRmdtdQEAc7wCEGtNm2BKWCf7ps5//xiCa9YxEwmUGJRhGGE4uksvKmpJetBybpN9bX9JG6+ES2VQm/KoljmwsLKMFBMEyEEfqWCV1hYNBEEBPX6uWOMzhZIQTDnupJSLiqgFiN65TaMzg6Qktobh6m+tn9BkeNNTFB+cTfW+vWo8Rixndsp735lURF4BmFZ2MdPhKJqYoEH1yBAXuZxVWfzrhNW85ABgR/AIiboBf6AVNuWxtP6m/BdB88OzeDLecKWvoddufg4JU01Q3eL9AmkT9WeZHjyAG2ZK4masxWGC+UBHK9KS2or44VjeP7sRaoq+sw93vcdcsVTbOy8nVSsi7GpQ0hmFyFdjUxbss6BEKRaNxFJd8x9XUoqU4OUxk+ip6JE1mQo7OvHnijNbOJOVaicGJsnrM6HwPUxmhIYmSiKpYfiSUpUU5+TaSgEIKDwWh/1kTe5B1ZL4wiF7JrtGLE3fT4pKY4dp5ofXqUdXwwCw4jjeXWC4MJdETLwZzJhfcBQo6xbcyvjE29w4sQP5oxtRDNkOq962wury4IgwJucnCMwgmoV6YYxgl4uR3CWcJKeh6xPZwnrOkJVL+hy8PIF/GIJLZVCmEaYobjQ9Gw7FEtSojc3YW3cQGXvvvMWQKuBMA0i27YidB3pulT2LSyqzmD39eMXCqjxGFo2g97Wdk5hJV2Xymv78HKLx9walmDTVRGsqEBKOP56nVJ+4QdMMyLo3mQy3OtQKS58DIWA5g6dti4dVRVMTXj0n7BXpDRaQ1hdQlQV2lpVmrIKmgb1uiQ3GTCRC+ZfTxLSKYU1nSqaBqWSZHDIw14gdjiVFHR2qJiGoFKVDAz51GrzlwVVhfY2lWxGQQiYyAWMjPpz9t3RrpKIC06e9simFVpbVVQFJibDbc9YeAPfoTw5QKp9K+IcxRCllFQLIytSx2pN07VkE+sp1cewnSKqopNJrENTDSaKx2e2K9fH6B19gQ0dt3HNxp8hX+4nCDwMPU480kLv6AuMF44iCRidep10rIutXffQktpC1c4hUIhaWRShcaj/MRx3/lPn2WhGlI5td5DumGvdlEHAyLFnKY2fRNFVFEvHq9hzLFDSC/AqF27NM1oSrPn4LpLbu5CBRE5/oZGuLPZIAfFmU5EEr1QH/62xFhmRJN077iOSapvzehB49O759mUprAwjzqZNH2Rw8CUKhUU7WZw3rltlauoExWL/nHi+BiuMlPiV6tyXfB8ZTFuoypW5IkbOXjdh2YILe5CWnndWsLUIx1qAoFKldvgIRkc7SixG08cewtqwnvLLr+CMjM6zlr2VqIkEemtLaE0rVxaPw5omcBy8QgFjTSdKJIKWOfcDYlCp4vQPLmmxNi2Fa26JseXqCFddH+W//mY/+55fOHmhe5PJ//6HnXzpc2PsfnLhtfrKXVF+6ffaMExBvRZwZF+Nf/hvY3jBxa+DDWF1iUjEBb/483EeejBCNqOgqgLfkwwO+/z3Py3yk2fOupAEbNui8ZEHM2zZpBONCmo1yaPfr/G5PytSKocngqLATbtM/tVvJti0QUPTwHHglT02/+1/FjnVO7twx2OCX/5UnA/fHyGbDoXVVD7gi1+t8PVHqlQq4Zg//9MxHvhAhL/7UpkPPxChp0sjGhFM5AL+9ktlvvqNCtMPdkwNHqSpeyeRVPuSVivPqTJx+lXc+tLiZDkUq8PEI61kYj1oSSO0WtVzHOl/nPHisZntpAwYmtxPzSnQkd1ONrEeRdFwvSrF6jDl+uxi4XhVjgx8n/bsDpoSG2hNbUMiqTsFJgrH5xUTXQgzkiaaal9yG+kFBI6HGtERmjLrXlMEirl0GvliKJbOmo/vovmOK+j/4vPk957GLdQgkGz57AMYTfELGnclsWJNmPHspZ7GeRGNNpNMrGF4BeICz8Zxyxw+8sh0fN87yw16WSGZH/R81jGfX79pmd+HoqBEo2jpFFpz03TZhQiKoYOmoZgmelPTMuYnKT77AnpzM5Ert4VlB957M7Grd1A/1Uvt8BFqR4+HVre3OCtOiURQE2HpDWFZZO7/wBzr3psRqorRPr32KQqKde4WRtJ18RdwkZ5NKe/z1T8bZ/0VFv/nn3Yvue3QaYe/++NRTh5c3H34nnsTOHbAn3x2mPFBF0UVeO7KXIMNYXWJuOkGk0//UpxnXqjz7cdqOLZkTafK+nUao2NzzVWWKfjYR6J87/E6//CVCtGI4BMfjfKrvxhn9x6b7z0enjxXbNX5w99PkS9I/uh/FMlN+WzdrPNrvxTn3/5uis/8/hTFokRR4JOfiPHpX4rzyKNVnn0+vEjuvy/C7/12kkpV8vC3qvh+aNXatlnjF38uxvd/WOd//VWZ5qzCb/xqgn/56wkOHHR59bXQbFYrjDJ85Gm6tt+DEc3ME1dyOj165Nhz5Pr3sxI3kqlyH8XqcJhoMG2J8QN3ugzD3PGl9JksnaRQGZjZXsoAP3AJ5FzTtuNV6R/bzVBuXxj4Thhb5fvOHNfgYsSya2aylxbDLdWwR4tEe5rQ01Hs0bDOjZawiHZfmPBQTI3Y5nbs0QLjTx7Cn7Z86dnYgqLqUoQ0xZt7ViRxYSGEUEnEO0gku1BVA8cpUyj0UquFQfwAppmipflKJnKH0VSTZKobXYviuBXy+ZPUalMz28bjHSQSnTRlt2AYCdrbriWVWguA59UYGdmL54XXn65HaWnZTqHQh20XyKQ3EIlmCQKfcnmYQqF3JnEiHu+kKbt5JuGjkO8lnz+1rHOrwYVwlgVqwbfP/0JQUyli111DZNsWjPb2sCSFqoYJTjNjCtCWl9Tj5/Pkvvlt4v39xK67Fr2tFSUeJ7rjKiJbN+PlclT2HaDy6mu4udxb5iYUuj7jwlSjkbAcwnL/drqY8bzg/zchg2BZgemBD74nFx1KUSCWVFFVOHmwTqU0/xhFYgqGKWjp1ClO+fieJBpXqJbfVA9MhWhMQdEEniOpVYNluwkbwuoS0daiYhiCZ1+w+dGT9ZmHEF0H/03nlxBw+KjHH32uMFNyYWzcZ9e1BjfvMvne43VMAz78wQjt7Sqf+f0cr+wJxc6zL9jEooLf+rUEX/qawbMv2LS1qvxvPxvjhZdt/vj/LVIshmMeOOhy3TUGn3goyo+erDM5FUzPSfCjn9T5n39Rol6XCAGWJfgPn0lx9Q59RlhJ6TPRtxe3XqJlww3EMl3TJRgUfKdGtTDM+KlXmBo6tGKB6yDxAwc/WH49peVuLwnmxGItH0G8ed05+8N5xTpTu0+x5mPX0/GR6xh/4g2QkLlxA9F1zfN0p1AVhK6iJSyEqoTV2RMWHnUC14dAIr0AN18hvrmdxJWd1Ppy6KkIzbdvw2iO40zMfSo8n1BBoakoujpTG0uN6GhxC992w2D45ZjQhUKied2qZLopik5Hx/V0rbk5jLsLPDQ9gufWOH7iezMuPMtMsXbtbVhWmmSqJ0xaQWCaKarVcY4e/TaV6hggyKTXk0qtIxZrRVE0YrFWDCMUqI5TZkw5MLN/XY/R3fVeTCOBYcRIpdaFBWu1yLS7b2C2551mEok2Y5kpUqm1DCjPzRFebwuEmP456zUpl3ceXPR+pv+RclnPZ8ub0fLnrTU30fTRj2Bt3YzQNKTn4U1O4eVy+OUKsm4TOA4gie+6Hi2dWta4fr5A4SfPUD1wkOjO7US3b8fobEcYBnpHB+nWVqJXXkHhyaeoHjj4lmTJnR0XFtRt3PHx86oltZzCqMB5PmcvvHEio/ILv9tK13oTMyL4wh+O8MarswlIuiH42K81sWVnhE3bw4rx//qP1xAE8M9fyLH7iRJSgmkJ3vfBJLc+kCKRUhkbcnnikTx7nimznETUhrC6ROzeY3O6z+Pf/k6KrZt1HvtBjTeOuJTL808Yx5E8+3x9Th2r/sGwrlVmOj4qHle46QaTUknS0qxw+y2zhd6khExGYcM6jedetNm8UWNtT/j/6682ZtZBIcI4ry2bdFIpZUZY5QsBL7xsU6/LmfGOn/QIAkgl58YMSN8jP3yY4thJjEgSRTcRCALfwakV8d23T2bHhaKZMaLpjnPXSJOSse8fQE9FaLptC023bsGvOVROjjPx1GGabtkys6mejrL+N+8i0pVBsXSM5gTZ924ivqWdwPXIv3KKvr9/Hr/qMPKdfaz71RSbfudevKpDYHuUjwwz9vhB0tetvaDPFOnOsvG370GNmqgxEy1h0fHR62m+YxuB7TH2w4OMfnffOccxIykiycULFF4MTU1bWLf2doaH9zI0vJsgcIlEmti48QNs3HAvB17/Mq57Js5G0NZ2NSdP/YjJyeNAQDa7hc2b7qe1bSenTz+JlD6DQy8zNPwKXWveQ1fXezl1+gny+emsNMk8ga4oOu3t1zI+cYiDb3wNz7PRNBMpgzmB6fn8KfL508Rirezc+alVOR4ry+zao2ZTmBvWYG7qQe9oRolHQUqCah1vfAr7eB/1Y334ucIFm0SFaaC1ZNA7WzB6OtDamlDikdBFHkiCSg0vV8AZHMXpHcYdGEXW35pitUokQuaD9xG56goAnL5+ik8/S/3UafxSeU5VcyViEdmyednCCgDfxx0bp/CjJym9tBuzu5vYNTuJbN2Cmkpiru0h+9CHQQgqe15b6Y83D+k4M9+jm8sx8ZWvh4VQl0nwplIUq0kp7/P1/zXBxistfuXft2FF567Bnit54pECz32vyC/+Xhu+J/mnz09g1wImRjykDL00tz6Y4qFfbuKRv84xfNrm2lvi/PJn2ygXfA7vPXfZo4awukQcO+HxW/96kp/6SJQHPxjhZz4W5dARly9+tcL3f1inVp89EX0fxnNzn2TPZNCesTprGmQzCmu7Vf7oP2fm6fm+AR/PD61NyYRCJCL4+ENR7r9vvrtqdMzn7PqUdVuSL8zdvx+ErYgX0w6B76xIcLpqxdCicez8+LJM33oyS+A6+LWLj9+6UKxYhkiiZVnbuvkqvX/zDGM/eB09E8OvOdT6ciimzuQLJ2bcg37VZuTR11CshWOv3MnKzOJV2NvLof/0CNaaDEJRcAtV6gNTqDGTyReO49uzN/hq3ySH/+ARqv1LV8B3JssMfOWlhYN4paQ+snTWzxkiqTaMyMUXhn0zmmbR2rIdx6kyMPgCjhNmWTpOhZHhV9m8+UES8U4mp2YTGvKFXkZG9s4EjU9MvEHXmpuJRVtRFR3P92cyQM/UNAsCF99f/AYuhKBWm6Kv7ykcJzwHFw9HkeG4l3lolZzuFShMg/gt15J4/43onS2LZrjJe27GHZ6g/PSrlJ/ZQ1Bcujr6DKqC1tpE9LptRHZsRl/TippKzHQaWGxuQbWOc7yf4vefo3boJHirG4Nkru0hsm0LQgjciRy5h7+5eBsYVQ1/LpCgVKb2xiFqR45gdHSQvu8eotuvRE0lSd7yXurHTuCXSuce6CIIanX8SgUtlUKJWEjXwy9duvV1KQIfxodcYgllwUR/KWG4N7x+K0Ufz5X0Hq1Tr85ehIm0yr2fSPPMdwv8+OEwU7r/hMO266Lc+kCqIawuZ6SEE2fiK5wAACAASURBVKc8/sefFfnS1yq872aTT348yn/5D2lMo8BXH67O234pfB9K5YDDxzx+97OTM9alsxkZC7MNqzWJ40g+/zclvvd4bd7Yrgf9A2eZmOWlKy+UWH8F2Z3voe/bf4t3DrGk6AYdt3+E6nAvuT1PXXSD5wsl1tSDoi3eGuLNSNenenoCTs8KUb/q4E7N3pACx6d4YJGifgvgTJTnuf0Cx5szJoRtafKvnjvLza845F89vez9L4RQVOJNPefMGr0QVNUgGm1GN6Js2HDPnHgaK5JGUfQZF94ZKuWROZl4vh/G5imKdlENpqvVCRxnmWLi7UAQgKqS+sgdJO+5GWGZS7pyhaKgd7aQ/uj70dubyX/jh/iFc9+I9c4WWn7jp9HXtISxSstwFwshUKMW1s7N6N1t5B95gsqze+cUrlxptJaw8TCA3deHM7x4D0U1FkONLB1ruSz8AGdgkKlHH8OY7v+nt7WhJhPnFFbyTcr9fE9tv1TCHc+hJpMzLXrcsaX7AK4+q1fuKN2skW3T2X5jjKa28EFW0wStnTp2fXmu+nexsBIIoVyyFGdNDTsbeB4MDft845tVTp32+MKfN/G+m815wupcVKqSfa+7PPiBCJ4Hh47MXVg0LdwXwOlej+ERn54ujf4BfyarEEILmBBveeLJoghFQZnugH4uAt+jeGwfTnHqEsaqCJIt86umNwBFNYg3rV2V+CqBMpOQoGmROU8CvucwPn4Q257rvvAXyu6c+bMLn2OYCHGZm6HOBwmxm3eQuH1XaLHwffxSFXdsEn+qCIFEiUfQ2prQMkmEFooiYRnE3rMTL5en8O2nzmlxDoqVMDNWVcPatEGAdL3QxTiRJyhVCGwHoalo2RRaaxNKzAq7YwBqJknqwdvwxiapHzyxaodDqNqMqyCo1Vk0olkIzLU9qMnEiu3bKxTxiiX01pawOfMi5Rvm4J/Ve1FRZhobL5egVsM+cRJrXQ9C04hddzW1I0cvaQmI1by+ziSLlvI++dzsffSpRwsMnV6eu/ldK6xi8TbSmQ0MDbx0ScTV3XdaJBIKx0+6FAoSw4DrrzWwTBgcPv/51GqSbz1a5d67LP7976X4678vMzLqz9TKWtOp8o1vVimWJH0DHg9/q8q/+IU4I6M+Tzxdp1yWxGOCTRs0hkd9nnrWvmzE1SzTAayLmc+kJH9k7yWt3q1bcSLJtnNv+C7EjGWwEs2rMnYgPRy3iuvWOHbsUfwFkiP8NxX2XLWz5B2kqQDUdILEnTciLIOgWqfywj7KL+zHG80R1MMAbaFraJkk0Zt2kHj/Taix0EoTug+vo/LifryRpVtB+YUylZcPYHS34U4WsI/3U3/jJPbJAYJyFem4SD9AKAJhmRg9HSTuuoHIjs0zDYe1lgyx912DfbwfuVCRvxUgLCzqhgHlzc0Iw5xvIRNh/7vkLe9d1GV6NkokghKxwsrwS6xfelMWPZsJXaDlyrkrsBPW6/KLoVVLsUyMrk7qJ04uf52Uksq+fUR3bkdvbyN6xRUkbrqB0osvIZ1FiuUKgRKLIVQVf7Eq8pcp+ZzHxIjLqcN1vvH5iZnDdI7Exjm8S4WVIJ3ZQHPrVYwMvYJ/CRTExg06v/KpGJom8Dw583z8+BN1vvS1C3Mj7N7j8H/8uyk+8ztJPvdHGZDhyRBIeGWPwz9/uwZIPA/+/C9LCAEPPRjl538mrOwsgFpd8qefLwGX8mnkzQjia7eSWH8FqhXDnhxhct/z2FOhOVqoGpmrbgzfNyymDu5m6tAriz9JriJWogUjNr/URAOIZTrRrZV7ej8b162RnzpFZ+cNxGPt5Cbnti3SNOuirJihKBMY+qWvA/ZWIzQVoakENZuprz9O+elXkM5cISFrNk6xgjM4RlCqkvnpe2fFTmsG66qNlM8hrAAqz7+GnytgnxrAG88vaDqXANU6tckCTu8Q2Z+/n+hNO0IrmaJgbVuP1pTCHVq6kOWF4gwO4ecLKK0tmOt6SN5+C+Xdr063qhEo8RjW+nUkb7sFva2VoFZDnKOWk9G1hqaPfxT71Glqx0/gjY3jV6th1p8QKKaF3t5K6o7bUDNpCAJqR4+ds6I5EMZAHj1GdPuVCF0n+b73ElRr1E+eCmPnVCUsqaDrYcbfAoLUGR6l8ORTZD90P2oiQeaBD2KuW0t1/wHc3GQ4jqIgLBM9m8FctxZz/TrKu1+l+PSzK1IaIpFWSTerdG0w0Q2FzrUGU+MepYJPfjwMPA+30ejaYGKYgo61Bj0jXmh9mvCWJYxKeZ/vf3WKj/96M1LC6cN1IjGFdVstXvpxiaP73sYxVopqYFlp6rUphFAwzCSKouK5tWmT/pv8xkJBN+JoWgQhFILAxXFKc59chYJhxDDNFJnsJjTNIpbomOmo7rpV7PrCXb9XAk2xMPUYilD5yld9nnq6TLbJxTIFricZGxMMDUbw3CYihkPdLfHM8za/8Ts5jp9QsPQUQeCiaxEK+TKf/Y81apUYmlLD9esEATz1rM3+A3l2XJGlKW3i+g6Do3mOnbApV2aPWaks+aPPFfnHr1fo7tKIRMKiowNDHgODsxXVv/GtKi/utjl5au5CeuSox2/8do6+/tUXpUYqS2rL1RSOvAZCkN1+E+23PkjfY19Eei7S9ymePEh9YpjOOx5CT6Tn9bl7SxAK8aZu1POIr3q3oKg6sWw3inphhU/PhZQ+wyOvEo+3s2XLh8hNbqNen0JRNCKRLAKFI0e/tbD7bxmUy0P4vk1Pz60YRpwgcJFSMjq2/7zHFEIlGm1C0yJEI82oqk4k0kQ6swHPrWI7ZWx79dahC0H6AeVn9lD6ye6lg8M9n/Ize4jeeBXmpp7wAUMIIldupPzjl8+5H3+ySOWFc2eXzmw/VaT4g+exrtyImgwfENVkDK29efWE1cgoxedfJHP/fSimSfruu4hff10Y6yQEaiKBlk4hXZfiM8+BH5C6+84lxwyLarZhtLcRv/lGpG3jV0LLGEKgRCzUWDzs8+n71A4fofiTp+cXPV0IKansP0Ds2qsx169Hb22h+Wc+jlcoIB0XoaoIw0BoKiN/8QWcvgViOYOA8qt7AUjfdw9aNkP8+muJXXs10rYJpgWaYpqz8XFShm15Vojrb49z2wNJrJhKfsLj9g+nuOnuBPtfrPDdL01i1yTX3hLjzofSWBGF4pTPrQ+kuOHOBK89V+G7X57EtWdvCqODLr4r5z1/Bz48+90C5bzPnQ+luPnuBPVqQO+ROvmJ5cXuXbbCKh5vZ+uVH6e/9ymSqZ6w4J9m4joVhgZeYmxk34wLT1UNunpuIdO0CV2PIRQVKQPKxUH6e5+hXBoCwDSTbNh0H9FYC5FoWKJ/25Wf4MwdeHz0AKdOPL4Kn0aQjfXQlbmGqJEJhZ/0GBw7wMHD+5AE6GqUtU272NTcjSI0AumTK5+if2wPQ8M1srG1XLXmPVSdKdKRNeTKpxg6rpCNrWVjaz/HRp/CDxw0xaLJvJHaaA9DYyqB9MmXTlCrvQLMvQiDAHr7fXqXEEcnTnqcODn/ZJrKB3Orw68iUgZM7HmaSv/xUDAJaNl1F1osiVvIARKvXMCvVfDtS1fOQVF1Es3rG9aqBdCMKLFM16oem3p9imPHH6W9/Toy6Q2kU2uRMsC2i+RyR2Yz/KRH3S7ge28+VySOU8Tz7XnWrXJ5hJOnfkR7+7V0dFxPEHiUSkOMjR+c/WvpU7cLuN7S8ZGGEWfjxg9gmSkQCr7vEI93sGnjB5BSkssd5uSpH67IMVkp/Kki5Wf2LCvjLihXsY/2Ym7oBjX8vrWmFMLQF3cdXQTu4BheLj8jrFAUtGwynIvt4E1OgaLMc5tJP8DLF8LvoFKd+yAWSPxiCTc3GZYWOLs2l+9TeuElhKqSuPlG1GQCLZtBa8qGzYIdB2dklPLLr1B+aXfoPtu5PYxJWuT4uRM5Kvtfx+xag4hYCE0LW8Gc8T8FAYFj4xeKVPcdoPTyK3gTy8+69gtFct/8Dun334m5Yd10q5npXqpBgPR8gmo1DPxdDM+j/MoevNwkiffejLVhPUo0gjBMVMsK2wD5PrJWI6hWsU/3UTt6bFHPgfR9vHweFAU/X1i6gCvw3PeKvPTj0rzoR8+TONPJWi88XuKVn8xPlPBcOUdUAXztz0Lh7djzn8A9D159usz+lyqoatib0PckrrO8p/XLVlgJoaDrEbp6bmFq8hi9p55EUTTa2q9h/cZ7sO0i+ckwdVrKAFU1KOR7qZRH8X2HRLKLNV03IWXAsSPfwffquE6Fgb7n0PUoaze8H0XROXH0uzP1Zc6kR680USPDxtZbsL0KR0afwHbLWHoSx6vMVFpek9lJU3wdp8ZfpFQfJWY2s6n1Vjy/Tv/UXoQQxIwm+if3UnMK9GSvpze3mxPjz7Gx5X1E9CQVO0dH+iqa4xs4NvY0VXuSVKSDja23UHEmGSseOcdML0+8SgknP06YngheNfyelItIY14NjEhy1Wo0vd0x402YsdVvY1Ov5+ntfYrBwZdRFJUzJQ08rz4jlsrlUV5//cszFdPPEAQeR49+B4mcV1IhCDzGx19nauo4iqKHvecCd84Y9Xqe11//yjmbNDtOicOHv7lInTN5UU2eVwMpJfapQdzh5VuA3KHp8iiqErroTANhGasirIKajX9WSYcz+wOoHz/ByOf/ChAEtbmC15uaYvzvvwyqgqzbc4ptBvU6uUe+jdA18HyC2lz3j6zXKf7kaWqHj2D2dKOm0whNRdoO7sQETv8Abi5sP2P39TPy//0lsaTGTbfpvPGsweTI7PllxVR23ADHf/gI+SCO3tqCmkqhRCJhfFYQENRreJNTOAODeJNTF1QY1OnrZ+LrD2Os6cTo7ECNx8LjYtv4pRLe+MQ5+wDi+9SPn8AZHEJvbQnHSadRDB3pefiVKt7UFO7oGF6+gFwiBsybyjP+D/8YBuH7AUFl6RAY1zm3sFnONmdYSFCdjZRMC7bzd31ctsLqDLZd5PSJH8+Y26uVMbZf8ylaWrdTLPQR+E7YvPXUE/j+bBuTqdxR4omwtYWmmvjTXelLxQE0PYrn1VHVgFKxf8m6NCtBNrYWXY1wePhHlOqj4edwZusGaYpFa2ILY6XjjJeOIwmouQXS0TW0JrcyUjgEgONXKVQHcbwqnekdTFZ6p6uIe2iqhaoYtKeuYKx0jFz5FCCxvTJtqStoTWxmvHjsbdkyQ/re3Eq/l2lwcCTVvmoxRG934tluNP3cPcNWAil93CWaZEvpz8sQPMNSzbWlDM4qMLrw+45z7sKJy93ussHzcftHzqsAZ1CpIeVs7KhQFcQqPgi9OS7oTFsZ6bqLxiGpiiQZqZMfc+bH3kg5HTO1xD49D2dgEGdgcOnJ+T5+vkAsa3H/r2xloq86R1il23Qe/PVOvvRfTjH16vzxDEshElep5NyLXvuCSoX60WPUjx4798ZLjVOrYff2Yff2ocaTqKaFkwtjXtVYHC2RWjBr0Ghuw2hqwa+UqQ/3n1eh0bcTl72wKuZ758QwVMqj1Ko5EqkuVFUnmBZFvu+g6zF0I4ai6CiKigz8MAV7iZTUt+IeHTUy2G4Je5FFW1ctdNWi5uTnCJ+KnaMtuQ1NDWN2pPQJpBf2rAscAunNNG8VCHQ1gqnF6EhdRVNsXTiICPdfrk9Mm5VX+cO+SxGKSjzbhfoWiYe3E4pqEMt2r0r9qgarj3Q93JHzK/Yr3+zyOtOW5gIRloliGQhDD3vPqUq4rk8LNjV+/rWiOjZEuPOTbXzt/+nFqV26B86Jfpu/+DfHmBhcOLRi241JNlwd59G/GMRbpjXmDMIwws4XzuqFbRjZZoxs84ywsjq6SF55LSOP/dN8y5oQWO1dmK0djHz36/jVd1C9t7O47Fe6N5vrpfTxPRvTTCCmm+NqmkVn13vINm8Ji/shEAJMK31Oa9RbEQ0TSC8UeIssLIH0kQSoYu7XoSr6tJiaveiXuqyk9MN4pPJJJitziz46Xu3t1YfsPBCqhmrFUE0TRdNRrSjGdAV2r1pe9exAoaiYseyq9cB7OyMUjUiylVims3Fs3qZIz5/jansrUKIWWnsz5uYezPVr0LIplHgEYRnTTYHV0AJ2xhKmLqOe0zS6IUi1GGy/JcXaK2N0b4ni1ANqFZ/JYZvAB80QZNoM8qMOkYRGPKPhu5KpUQdnukikGVFINusYloLvSYo5l2pxrqCMZzSSTTquHaAZypwFXNEEzZ0GZkQlCOS8e5FhKWTaDK6+M0OqRad7axTPkZTz3sJWtgWIb74K6XmUjxw498YXSK3vJLW+k8va1hkfoXzsDYymlQ+Z0NqaiN96PVpLFm98kspL+3H7Fy/euppc9sJKVecWMxNCQVF1/MCbFgqCto7r6Fr7PkaGXmVi7CB2vYgfOGza8gCp9LpLMu+zKdZGaEttIx3tYrx0jEAGKNOiMJAejl+lbE+QjnYxVjqG69fQFINMrIeSPbbsRsCOX6NUH0NVDKaqA3i+jUCgKjqB9FmOuSrbczXxbDerKjllQH7kKMXRc5ujA9fBq5TmiELpu7iV4kywo5lppem629BjSVQrQqxrA0amGbeUZ+zFH+BVVqblgxAKimagqDqaGceKZ7ESLViJZqx4M7Fs13IGIdGygZ5rPrQic1oMuzLF2IkXkcHqN2mFMNNN0XQUzcCwkpjxJqzpn0iqDTN+7vgqIRSyXTsw402rOtdqfohc72uXrDjw2w4pVyU2aiGEZRK5eguxm3ZgbuoJxZQWVsJfKWHeutbizp9tY8uuJNkOk4f+VTeBLzn9epnv/80wtbJPc6fJz//H9bzyeI6tu5K09ljYVZ+H/6Sfk/vK6Ibgg7/ayZYbkmi6QAjBeH+dhz/XR27IAQHrrorxwKfXkG4xqBRdRk/XMSKzAtCMKNz0QDObrkvQvi7CX33mGMf3zno1urZGueuTbWy9IYmqCz76293IAPb9ZIqn/2kMz118PReGQXzzVaR23kDg2BjNrdQHe6n2nkDoOtG1mzCb2/BrVSonDuOVCqixBNF1m/CKeazOboJ6jdKh/aAoxDZuwy1MYnV0g5SUDu4lcB1iG7dhtnZgj43MEW9C10ls24mWTGOPDlHtPb6qDaOVeJT0J+4jet2V0zFbPsa6TnJfeBg/v7otfxbishdW8UQHQtFmbhCGmcS0UlRKwwSBixBhTSrXrTLU/yL1+hQQNkPVjdii5mcpg7Cr/bka5a4AU9V+pir9rG95D8lIB45XwdRiVOwcQ4WDSOkzMLmXTa23s6n1Nsr2ODGziaiR4fjoU3jB8sy4gfTom3yVre3vZ2vbXVScHIpQiRgZhvOvz7NiLUS6fSst63et6nEJgtDquBxhVTp1iMrQKfz6bGxLdaSPgR98ZSaI3c6PM/bCD+Z/1zLAq13Ik3ZYD0coKpoRxYimMaPpUCgkW7HizWGDac1A1UyEsrz2GxDeHOLZLuLLEWEXQWniNOOndq+CsJo9NroZw4hlZ49NogUr0YJuxVE1A0U1zvPYKKTaN5Nq37zCc55Lrm8fk/3758btNViCt6anlZKIkvrQ7cRvuQ4lEZ05b2QgwfMIXJ+gUiOo1gjqDtL1wh/Px9zcg5ZeXnzjWG+d73x+kFs/5nLd3Rm++J9PYdd8PEdSr4TnhFAFLd0mO25N8/Q3xhg5WSee1hjrDx9yPU9y+mCF/U/nKU64NHeZ/Ny/W8f2W9M89bUxonGVez7VgaIK/u73T4CEOz/ZRjQxe8utl31+9MUR3nihwC//4UYUde51MnCkysOf6+enfrcbI6Ly9T/uxbUDnFqwpKgCIAjwSkUCp45bzGOPDOCVwgzIxLadRHs2UD19ArO1HaP5dnLPPI4ajZG54RbKR1/HKxaQfrh2KKZF+rqbqfWfDt190+eC9AOciTGsjm5i6zfPEVZ6ugk1liCo10hf956w9MLJ1Uue0tubw/Ie2nQcn6ZhbuxBa2tqCKuFSKbX0tZ+DYX8KYRQae+8HtNI0Df+JL4XPhm4bhVNs4gnOggCF0XVaW65iniicybF+myCwMOuF0gmu0hnN1EqDoQlEHx3pnnrSuL6NY6PPkVzYiPZ2Now5sorU3MLM5aYqUofR0Z+RFtyG9nYWupuicPDP6RYGwZCV95UZQApfTy/zlR1AN93CAjIVwdw/TBrJV8d5I2h79OeuoJUpJNAepTrE1TspZvsXq4Erk3gzhWW0nPxyoU5v7ulqYvel2aGVdOteJZIqo1oqh0jmkEzIqi6haLq71p3lh5JEU22YSaaiCTbiCZbMaLpmWMjlnB1N3ibc2GJUeeFsExSH7qD5L03z1Qql1ISFMrUDp2k/sZJ3IFR/HI1LE7q+6Ew9gOkhNbf/uSyhZXrSAoTLrWSh+dKChMOdnV+uEDgSQ48nWf/U3mQMHZWn2UZwMHn8rSti9DSZWLFVaolj6aOMB423Waw7qoYD/9JP4PHwrV59w8mufrOzOwYEmpln3LeIwjmH2CnHuDaDnYtAAGFcWdeyYDFkJ5HfagPN38FzsQolROhqFEjMeKbr8SZGMO3a7iFKZI7dqHFw/IU0vcoHnwNrzC7niqRKAQBleOHqA2cnjvH3BjOxCiRrnVzXveKUxT27Sao11BjCSLd61dVWImoNb/CvaqixFagT+MFcNkLq8mJI9OuvltQFR2hqAz0P8/E+EHOpN8PD75MPNHJpm0fxnNrSOlTq04yPPAyLe07540Z+A6jw3uIJzrZvO3D0xmDPqMjr9F/+qlV+RyOX2Uof4Ch/MK+bomkUBuiUBta8P1SfXRORuHRkR/PvHd09Mk5I529bYPl09S9k64dH0AzIg2R8CZaN97EmivufMcKqPabu+m6Yz37//wlnNLl1HXgXYAQRK/dRuKOXbOiyg+oHzpJ/uEfYZ8cWLq+0pn2EiuMY4cxVQuJykybwSf+TQ+JjE5u2Ma1AxJZHaEAAmJJDU0XTI3MnkulSfe8g89XHFVBtSKokShmS9h6q3jgFfx6FTUaJ7DteWUpAALXPUdIxdzPFbjutMVLEjg2Wjxxfj1hzpOgWJnODI3Nzsi28afeemsVvA2EVbk0TO/JJ4jGW1EVg7qdp1Ka25W+VBzg4P4vEYu1oag6nlulVBpCUTSKhb4FO80XC328sf/LxBLtqIqO7ztUK6tTqbfB2wNF01G1d69VailUzXjHiioAKxshu60FRV/90IAGc1GiFrGbdyAisx0LnN4hJr/8Xdz+ZTwgCsJ6U+eDnJYCYqloUrmgDlBUuOauDD1XxPjcpw8zOWJjxVRaumYzgqslD8+TpFoMILz/RBMqqn5h10943Z1fOwkZBEjPQ4snUSMxAs8lsG3qI4ME9TqF/a+Gn11R8GuhsJr+1Mufl6aFYljREJo+4z7UUxnMlna8SgmzpY36yCBIiVBVFF2fbqNjgFJbkXY37vA41VdeJ37HDQjTIKjUKD+3F3do7ILGC0MY1JmqA+fLZS+sEALbLmDbS/dEsuv5ee1ofGAyd3Txv1nGuA0aNGjQYPVQU3H0rrbZmCrfp7rn8P/P3ntHx5Xdd56f+0LlKgBVyIEEcw5Ndu5mB7VaHSS5JVttW7Zsy0mTPLuzO/KxveOzM2dmd4937TNzfDRnPWvLkiVZ0Urd7ZZa6szO3cwJJEGQyLGqULnqxbt/PLAAECAJgGQzCN9zGPDqZbx693t/4ftdsCWN0HVEYPE2UqW8Q7hGo219iOSggW1JSrmF+cmJKY6j+wU19Tob7ojRsibIwGmPRGXGLYZOl7j7E/UMnynhurDjoTpC0WktL1UX+PwKoaiKogqCEZVQVMUy3WrKT0ooZmza1gZpWR0gO2FhGi7l/ALqA12HUl8PNTvvpP7Bxyh0n6B49hTZwx9Qs/NOGh58DOm6GGPDZI/uA9fBKZfmRpVcF6dcnKOMrsVqiW3dRaC5DTUcIXHfI+RPHsG1LMx0kujGbagRr86q0H0CoenU7LiDQEs7WiRG3R33Ux7spXDyyjsWpWGSfe41jHNDaHUxrPE0xqlzSzbijtS14wvUkBo6sqTtb3xitYwPFVJ6harnZy23aoRiGcu4EBIIJEIktjahhX2UxwukTozjlKfrNNWARnxjA+GWKFbZInM6SXEk7w2yYR/xrY2URgvUrk2Q653EzBsktjRRSZdIHR9H2t7gpAV1atcnCLfEsMsW6a5xyhPFX0idOTUWQfFPd3+7FRNraOzS6b8Z0OI1qJHQoo975mCewdMlfvPPV1EpOZx6L8cLXx3GKLlI17NAcZ25vxDXgRPvZNm2p5bP/5c1VIoO6VGDE+9kcSwvFFbK2bz4jVE+8S/a+MP/Zx3FrM3gqSIT/Ua1nmr97ih3fbyeeLOPaK3O47/Xyp1Pmhx7I8N7zyergZxDr02yckuY3/nPqzFKLu89n+TNH07g2Jd/WEr9PRjJMRRNxykXQUrM5BipN15EDYa8OrZKGWlZmJk0yVd/Msd/0C7mmXj1pzil2TqMTqlA/sQhr3Pw/LJiHum6mOMjnsehz49TLuEaFRBiitxNBzuupr6WWyhReneuN6GqB4kmVqLpAWyrQiHVh22VPf/Sunb8wVrMSo5Cuh8pXSJ1HTSs3I2qBVA0nUohTSHdtyi5omVitYxZyI11IwDNH0LV/J68wJTEgKr6vJb6RXZ7LWMZNwN8UT9b/uB2hCJQfSrBxgh9L5ym6xuHwJXoUT8bP7eDxl1tGJNl1IAGUtL1jUOM7xvCHw+y7Qt3kjuXJtwWQwhB9myaSFsN/toA7/8fr5LpTuGL+tn42ztp2NGCkamghXQcw+bY3+1j8uQvYDmCcoF4qOPgLjTSIAS+1W2oCyxcn4nkoME3/tM56pp9CAH5tF3VqEoNG3z1P/R4NVbzYPRcha/8hx5qG324jiQzbqIowtOqwgv6dB/I8/f/Ww+1DTpm2SUzYfLGDyfITnj7HOou8+q3pA7o9gAAIABJREFUx7iwATuftmZlx84dKfCV/9BDTb2OlJAZNxdEqgAv2lTIcWF8yzUqHtmZCcfBLsyjhD61jwshbRsrM39TlGN75GyWAKiU2LkP31y8dd0DaL4glWKaYKyJSiGJbRs0dt5JuLaVcn6CuuaNBCMNTPQf8MY6LTCdfl3CMHfDEivHMcjnh7CM61N89ouKVP8hUv2HvB+EgjpFpBRVn/6jeQ+e5gui+UJo/hCaL4w/XEtN83oUVb9u5x8KK7R26qTGbCaTc8PldfUq4ajKcJ85J7VfySdJDx6f8phbHBRNJ9qwClW7dEpCSolRSFHKjCz6GItBOT9xVYVRS5lRJgePspS3jOoLEG1Yfdn7KqWklBnGKFzbDtZCqn/e2WcgEWTsW4Oc++eTIATrnt7KikfX0vv8KcqpEm0PdNJ670r2/+UbJI+O4ov42fi5nWz+3V1ke7xzVnSF5NExur5+kPv/6gmKb+bp+tpBbv/TB6hdlyDTnaLjkTU03tbKoS+9Q/LoGP7aALu/uId1T2/l4H99C6t4bS22bjS4ZWOW9IXQ9Skfu8tDq68l+tAdi6+xmkIhY1PIzO0ctwzJcE95ni2mkU/b5NOXkDORkEta5JLTEaBKcZrM5FIWudTl9cGkhOyERXbixvKQvFkgFBXXschNnKGcT+I6Jr5gDfHWzUz0H6Ccn8CxyiQ6dpIcOERuoodwbRsAE337lnTMG5ZYFfIjHDv0jVtWLfymgHRxLAPHulS49jyjFwRjDWyKd+ALXj9itWaLnz/761a+8/+mePYbs2dHuk/w6/8qwZbbg/znfz3E+NDsl+Lk0Akyw10gBMGQwHGozmAvB1+4jk0PfQE1cplaDymZHOmi/+Bzi7quxULCVSVWyb79pPoOLGnbUF0bmx7+Aopy6dZnKV3Gz77P+Jl3lnSchUJeRJepPF5k9L0BHMMb5DPdKVY+vh4t7EMrWdRvbybXN0nq+BhIMPMGw2/30f6R1YRbo5g5A8ewKQxmqUyWMbMVcn0ZrKKJmTPQwz4UXaFhdyuliSJOxSbaUQNAcSRH4+42fDX+Xzhi5WTzOJk8Wty7F8Lvw7+xk9KBLmTl4u8eNVFDzVMP41/X8WGd6jJuQox07yXeupXmNffh2AYj3W+iqCq6P0I0vpJQtAmEIDN66qp56d6wxApYVkZeKgSE22qIb2shUB9Gui6FgSwT7/XjGFdbMPJ8e42cU9x4PSAE6LqYI7YHU4WgeZds2rlI27NESomqwCc/F+fMcYP9by6s7sW79gWG56W8+SYMM+ruFr+ps8BbI6/rvbHLNvaMeirpeOcjhEBRFfSwjpU3veVTsAqelp4e8mHmDKQE13LBBWm7yKk6ISnlVM2Jih72Ubsuwe4/3jPl9QlaQMMqWnw4Jls3FtxCGeN0H77ONoQiEIogtGsT1sAoxXeO4JYq051jqoLi9+Fb3Ubso/cQ2LYWFBXXsFD8V3lCpyieDpIy43diO975XKFsgAgGvCjb+V1LcEtluNBjcUHnGEBrTKA3JlBqwig+HxKJLBk4mRzWaBI7lUVWKldWwydACQXhvAin6+IWK3O7+qbum97SgN5cjxIJIXQV6bi4xTJ2KoM9lsLJFT2j5oVKZVx4/Mth6nclpUNy8DDZiR7a1j9IrGE1mdGTlAspMuPd5JPnqpu4tglTmpZ6IIqi+pDSXbTQ8g1NrJaxNIRaYuz444cItcaopEpIVxIdzpHcN3C9T+26wrYkP/xKGn9QmTdNeB6hqMp9j0VJTziLbHC+sSBQqBEJ/ASoUCIr01zLqwn66rCc8oItmG44yIsrjLu2g1Uw0aM+hCqq5EqPehFKq2zO2s3F4Foudslk4sAwXV8/OGuiIx1JaeLWNKW9FKRpUdx3gtDtW9ASNSAEak2E2l/5KIEtazBO9eLkSwhVQa2J4FvVjn9tB2pNBBBYg6OUj3YTe3LPVa37VGsjxH/7KfTm+uoyc3CMyX987srUvFWVul97nMDGVdVFbtkg9ZUfLtzbTgjUeIzgjo2Ebt+C3tqIEvAjfLpn6SIlOC6uYeIWy5i9QxTfPYLR1eMRwyVACQao+42P41/jRQjtyRzprz2DPcOgW62NEty1mfAdW9FaG1GCfo9AKt45SctGGiZuoYw5OErxnUOU959Y0vEvB7N3mMlv/5SWtrvRfCGkdFFUnXJuDMsokhw4SLxlC7H61QAUJgdIDhwE6VLMDNO24SE6Nn+MfLqPyZETSHfhpHeZWN2CaLijg9jaek599QOGXz3jjaUC7NLNkaNXNWjp8FFbr4KE5JjN2JA1J7MlBNQ3azS26iBgbHD+6/MFBGs2+QmEvCrRSsklN+lgXRC1isQU6pt1Nt0WoKVDZ+U6H7fdF6qOt93HKhRzN0+kKS6a6FDWkZVJ3GtsSSJQWJHYzUTuDOli7zU91vWAXbGZODTCht/YQWJrE8kjXo1V670rKE8UKQ7n0cO+y+7HNR3G9w/T+cR6/LVBkke9gdRfEwBBNcL1iwbjTD/Z5/dS92uPIfw+hBCokRDh27cQvn1LNbI3kzhJ18U8O0j62z9FVgwiD+xGjS6sNmshcAtl7HSW4G2bqsdVYhF8a1ZQ3n98yfvVm+IEd2xAS9QCU3WX3X3YyYW5RwhdI7hrE9GP3Ye/s23++jIhQFFQdQ01EkJvShDcvp7ykdPknt+L2Te8+KiboqDGa9FbPQNlJRRErY1ViZVvdTu1v/IogQ2rPII3zzkJvw/8PtRYBK0pjj0xuWBihaKgJaaPfzk4hZI3Lpx7D90fBSRWJY9ZyQOS9NAxCukBdH8YKV2MUqZ6T/LpXnoPP4vqC2JV8ovOxiwTq1sQoeYoVtEkfWQYIzVXRfdGRrRW4df/VYK7PxJB1QUCz4Lip9/N8vy3JqmUvQdfUeDBT8T41S/EidWpGGWXTNrh8DulOZaB0RqVX/69OB1rfDQ060wmbf7ktwZIjc0O7973WJSPPBWjqV0nUqPyyKdquOeRCBJwbfjLPx6m+9jNo8pdIxKk5CgDbjdXEqlShOalFqSDYH5/TSEU/Hr0Q/HevC6QMPxGH5H2Gnb80d0YkxXUgNcZe+IfDmCkywsiVgD9L50h2BBm27+8E6vkKXsrusLIW/2c/t7RqiTDLxRsh/yr+5CGRezJ+9Gb66d937iAUE1Z3RT3nyD3wlvYI0mUcBBrNIWyBNmFi0GaFpXDpwjftb1K2JRQkOCO9VSOnl6aMbUQ+DevQa2rmXEgSXnfcWT58u8W4fdR88mHiHzkrkVLTCjBAKE7t6F3NJP53guUD568opSm0DXUmHdf/Bs6Sfzup9FaGhYcNZSGReX4mYUfcMoMXFo2aAvtSpeY5RxmeT69SolZzmCW5+lUlJJKMXVe23XRuGWIlR5Uka7ENm6Ol1LbR9ehxwIMv9xNbE2C+ts70II65bE8I3vPUhqebm/Vo34SO1qJb21GDWgUh3JM7Bug0DfpGZQCkRW1NN3bSbAxQv2udnyxAOt++3asnPdlHXq5m+SBwer4qvhUEttbiO9oxRcLYEyWmfign8ypiVkv9mBzlM6ntjD489O4pkPj3SsId9TilCwm9g+SOjR81WbZPr/gl38vzgNPxPj+l1N0Haygqh7hefoLcTIpm1efy+E6sHKdn8//+3qGzpl89a8mSI3brNro5+k/iKP7Zw/uqXGbv/zjEaIxlc9/sZ5NO+cvon7rZ3kOvlVky+4g//Y/N/P9v0vz5gu56rvnUunDqwEFhXrRSlh4L928zJCSw8RFEwJBUo4CklaxipxMU6JAo2inQomYiKOiMu4OYmHSqLSTEM1YmOiKj0k5zqQcJyYS1IkGVHQMSoy5/dhYqKjUiSaiog6BICtTpOUoQiisbrwPR9qcG3+L5tpNNMY2zD15IagNtTGUnqsjczMgeXQMq2jOKhzP9aY5/uV9lJPe29Uqmpz8xkHG3h8k3OrpT2W6U56OFWCky5z8+kEKg1kcw6brHw+ROZPCqdicffYkxqTXZWblTbq+dpChN3q94nUhqCSLZHvSNwSpslNZJr/7M0TAI4rStLDGU4vahzkwSvqrz3hpKcAtVXCLl5nk2TaFNw9gdPcT2Lwa/5p2tOaEV1cDuBUDJ53D7B+l0nUWs3cIaXjkxi2WyfzgJbT6WnAlZu/QIq96fhjnhjD7RwhuWQuAUASBDatQ6+uwl6DqLQI+QrdtRsyo23IyecrHFkAwVJXoY/cRfey+WREh1zSxR1OYvUNYYyncUgWhqqh1UXwrWvCtaEGJRRBCIIRAb2mg7rMfxy2WMU71LvoaqtcyRaz0tibqfv1JtJaGqfOxcLMFrPEUTiqDNC2EpqHURNCaEqg1EZRgAGs8hTWycHkRt2KQffZVCm8dRAn4UEIBlHDI+zcaJrBl7ZL0zK4FbhlitemxDgoTZc6+dXN45MW3tVCzrh497KP5vlXYFRNFV4lvbSZ9dLRKrPzxEOt+ezeNd3RQGs3jmg6Jna20PryGk3//Psn9gwCoQZ1AIuxp60xBCDH9BZ5B7hWfyupf2c6KT2yikiphFQxqNzTQ9shaTn31A0ZeP1slS76aAO2PbcC1XGKrE/jjQRzDxh8P4Rg2qcPzexsuBS0rdB54Mso7L+f5yXey1VTd2JDF7gfCPPBklPdeKVAsuOzaEyYcUXnm65N88HoRKeHcKYOVa/188nMXhKElmBVJznWolC4+QyvkXAo5l9aVDq4LuYzD+PDClJivBkLEaFVWM+L2Ti3xCsYjohaBQkqOIoFapRHTNajIEgmlBReHtDuKi8TBwcEmJ9PUigYMWSYtx6hIjxxo6BiyjEuRJqUDS5iMywEaRDsNSjtJdwgHBxd76gzAciq40vs57G8g4KshW5r9exeIeQ3PbxYUBrIUBmbPassTJQZfOzdrmWM4JI+MkjwytxbGKpoMv9lX/XnkrWnX3okDs++XY9hMdk0w2XXj6Va5hRLFd5emOH0eTipL4Y0ldJG6EmtkAmssSeGtQwhdQ0yRM+m6XkGyac1b5F1ZCDlZ7OkUS5SPnCKwaTVC8c5DjdcQ2LiKwhKIla+9GV9Hc/Vn6UqMM/3YlyOuQhDctpbYo/cgfJ7tlpQSJ50l/9K7lPYdw8nmvWiO43ppN01FBP3413QQe/x+/BtWIRQFIQRaY5yaTz5Mcvh7uPmlhmU0tJYGYp1t+Fa1geti9o1Qeu8I5eNncNJZpG17xelT56OEgvg6W/FvWIWTK+BkC5c/znk4LsaZfjgz9b1SvHSnUBWUcIiGf/uby8TqaqN+beyma6iJdsa9me2X3yV/Lu3NJiI+SqNThZGKoP2xDTTft4qe7xxk+NUeXNsltjrO5n99L+s+t4vs6QmsvEGuO0lXbxqBYOMf3kXzntV0/+N+8mc9fR3Xnu7MSuxoZdXT2xn4yUn6njuOXbbw1QTZ8kf3se5zu8n1pCj0Tef7taBO2yNr6XvuBCN7z+JUbLSwjlOxr+oMu2ONj0SjhqII7vtYpJq8mioXoK3TRyCkUC65rNnkJ5u26T1tVomP68CpI2WesGuv2jl9mHCwkUh8ws+kO06R+QtkL3zMM26SMTm7MSEn01RkkbIskpHTg3depgmLGnT8ONiERRRVatSJRlLuCCOyd/bOpUN/6oPqsyOlw3j2JL3J9y44JxVdDbCMZVwVuBJZMS4pt/ChwHExTpzFyebRptJ3wqcT3L6e4psHFpcOVFX8G1ehxGYYBZsm5cOnLrsftS5G9LH7UWMROE+qJrNMfucnlPYdn6tSf75Q3LIpHzyJPZYi8YdP41vdXo1cBTatJnjbJopv7l+SibVQBOE7tyH8fk9tfv8Jsj9+2bMimqcmSRpeZNGeSFM+dBJUFewrmIy5ElwHaTu4qnFDdKWfxw1PrDp216P5VPo+GCfaFKR1a3ze9RrW1ZA8O49q7A0MiaT/+S6S+waryyrJ6dmDvzZI8/2d5HpSDL54uprWSx8dYeztXtZ+9jbC7TVkusaRrkRO6e9IxwUpcUxnjryCGtBo3rMKp2zR99xxymPejMHKGQw838Wu//godVuaZhEroSpMdo3R99wJ7KlUiXENNBxDERXdp3D3IxF23DN35jE+bCEUj2iFowqWKSkXZ3+ZClm3mh692VCmQLdziITSQqe6mbycpN89BcwmU4LpVKdEYnBpIcPz0PGxTr2NiixRkrmpngYFgUARKpacfxCbKX+QLvbhuvY8kgiSkpmpRraWsYxbBebQGMaZAbQ7poiVEPjXrkBrrsfqX7jQrxoNEdyxoRr5ArBGkxgnz11iK2BKfsK/bmVVoV4aJvkX3pqfVM0Da3iC7DOvkPjC09WojtA1Int2Uz50Eje3iMjRzGuqiSKlpHKse1HdktKywbp13xU3PLFa/9E2gjU+ho6mWHFHIx/599vIj88dSCINQU7+fHCePdy4cCo2uTMXDwHrUT+h5hiySbLrzx+tdsaAV6CuBnV8NZcWXbwQiq4S7Yzjqwux/YsPzSIhvpgfRVfx14W8MOuMz/Jn01VSda1QKjhYpst3/0eKd14qzPUCdSTZSQchPD0q3ScIhhRyk9NpgVBEuWmtdjQ8/Zkxt5+KKNKqrGaYcxiyQlxpJEgEBZWQWLx9B4CPAAFC9LkncXFI4KUkHGzKskBcNFGQGdwpkbz5CNtksX/OMvAIXl/yfRz3Jug81VSUgB+3cHM1dixjBoRAq6tF2g5O7hpPqG2H0vtHCO3ahFC9gnolGia4ff2iiJV/3cpZHW3Scakc7cZOz1dYPQ01FiF05zaUGXVV5rkhiu8fXbCfIoBxuhfjTL9H7qbekXpbI/7V7V4EaYlwMnmyP375yiQobjHc8MTq7f+vC6EKzKINUnL6lWH2funYnPUe/J+23nSRCulKXOfiBdFCEQhVYGUqlEZzzBSFLY/kSR0cpjy+yJmGAEVTcA2b8kh+1j0rj+TJnkqS7U7O6RZxP4Si2v5uk4kRmw07grz0o9ysaJSqevfDdUAocOZ4hTsfjrB6s5/xYQspQVFh484g2hXqBDq2xHUl4ajChylkFRRhVijrPWIjYdwdxMZkUo5RQ5xV6mZMaVCQmWra0JJGtR7qQliY2Myw06BEViZZoa7HkiYVWcakgkQy7J6jTVnDanUrrnTJyhQjsndRSsSWc3MQlcDaFYRu38Tk915cWmfXMq4NhED4fJ5o5GWgxqIkfuNXsZMp0j9+7pqnDI0zA9ijKbTW6a634I4N5F96d0HHFj6dwOa11UJ8ACdX8LriLjEGAOgdLbMJmetSOdGDk10ckXFLFcy+YYJb11VFNpVwEF9HC+Ujp+dN310O58/F7L16tba3Am54YlXOTEdJCskKw0fTFJNzH+T8eAX3FtOAscsWZqZMoT/D6X/Yh12aGzFyzMV1qknbpTxeQAv56P7WAczM3KiEa7nXRRVzZMDixR9keep36vid/6WeI++VKJdcwlGFFWv9dB+rsH9vEdeF/W8U+cRv1vHpz9eh64LkmM3KtT5uuzc0J9LlDwoCQYVwzPuj6YLGVg3XlZgVSanoztLISo7ZZFIO9z8WZbjPJJ9x8PkVzhyvUMxfu2esIDN0O4cRCCQuFlY11dfjHEVFw8VBInFxcXHodU/gXIRYDbk9s9TSHWx63S409Or251GhSK97Ag2PldpYi7Z3CPsTmHYJy1lYavK6QFXwr+vA19k62/h3GdcdWiJO5K47yL706oLIFY6LtBeq6n9lcCazVE6dI9JS7xFAIdDbmvCtaMY43XfZ7dVEDf4NndPNRFJi9g1jDlw+4uVf3YYSmq5fdEsVzP6RRUWrzsMeTyMtuyplIYRAa04gfPqSyKk0LSpdZ73U3jKquOGJ1UwM7Esi5rEqATjz+ghm8daafRrpEuljozTs7qBmXT3j7/dXXyJCVdCj/qqv2ULhVGyS+wdpuL2dxM5Whl7qni5AF14XoGtfH68yy5Q8980MCHj4kzH2PB5FUQWOI8lNOvSeNqrv0IEek6/9twme/sM4//LPGzEqkvEhi59PEbOZ+Oy/9nSxdJ+grl7DH1T4k//aimlIxoctvvS/j80SFx0ftnjma5P8yh/E+Xf/VzOWIcllHP7yiyMU89fu3kgkJvOrIttYs6JP52Fx8fOZb30H+6JE7FKfXQ4Chbb4bSRz3aSLlx9oFgI1HkNvSiCCfnBc7HTWK4ydb0BRFbSGOFp9LYpPw62Y2OksdjLjdZBpKnpTAr05QXD7OtRYmNDujUjTu147lcE8d0GnYzCAr6MRJRxCGgbWcHJuukNR8K9bgZsvYo0m0Rrj6I1xUAR26hLnu4w58LW3Edy8kdxre7lIuV8VTi5P8jv/NKXk/SEUuLuS0sEuwnfvQEyRHCUcJLBlLUbP4GWjToH1nZ4UxBSk41I5fAq3eOlJiPBpaM0NXsj+/LYVA+cy6cOLXkapPEdBXK2NInRtacTKsLAWQA5/0XBTESv7EiRi5Og1qKa+znBNh7PfP0KoOcb2Lz5E6uAQpbE8WthHpKMWM1PmyF+9vij/P+lKhl/roW5LM5u/cA/N962iMJBBqIJwWw2KrnLkr16jcp2sNcpFl3/6uzSvPJOjrl5F0wWWIUknHTJJuxpZcl3Y+3yeY++XSTR5j/HEiEUh53J8X5nMDNf5l36Y44PX5r8ey5RMJmffP8eGn34vw/43i9QmvBdaqeAy0n9rEfeFwpNpvXR0RwiFgB5FiAX6eF16Z4Tv2krtpx/29HpcF6F5vmqFNw6SeXYv0pgmlGpdjJpP7iG0Y0N1fVQFJ50j+ZVnMHuH0epi1H3mEbT6Ws+iRFGo/aUHqynv4v6uaWIlBP71K6h96iH0lnqPGCkKbqFE9qdvUXz/WJUsCb9O3dMfxeofxRpLEXlgF0oogPDpWCNJxv/620tvZ78KELqOloijBDwy4BoGTiaLW547oCuRCFptDULTcCsV7FRqViRCSySQrou0LLS6WpxCESeTQa2tQY1GcbJZnOzseieh62jxOpRgEGnb2JOTs/WsFAU1GkWtiRHavhUlEMDfuRK37E0w3HIZa2Ra2kIJBtGbGr02YcApFHCLxVmlC0owiFpXi51KoyXiICV2MgVCoDc1eLpcydRlydCFMM8NYvYNe1Y0U1Gr4M6NFF7fd0miI/w+gjs3eqrjU7BTGSonei7bjScCAbR4zay6USUWIf57v4y0Fv8+UsJBlMBso3glGJhVUL8YuMXScq3iPLjhiVWwzofmW9jLupIzsco3h3FzZaJAoTeNe5lUXqF3kkP/9yu0PLCahjtX0LQqjlOxKQxMMvpmL441j55Lski+bxL3IoTLyhsc+9KbtDy4msa7V9Kwuw3pQmkkx8jrPdXuQ/D0dnI9qXlThtcK0oXkqE1y9NKEUUpP/DM1Pnu9sydnz7wGz5kMXqbx5kK4DowOWIwO3Jhkau1ddWx+sIF3vjfIRO+1e7EpQmN9yyOEfHWXXE8IQTTQdHUEQqXEGp4g97N3MHqHcQtllHCAmifuI/rRuyifOEfleI93XJ9O3a8+SmjHevKv76d04CRuuYISCaFGQlijXnOIncqQ/PKPUSJB6n/3KUTAx/iXvlMlaHLG90hvbSDxm08iHYfk3z+DPZZCrQkTfeRO4r/xOE6uWD3+eQR3b0LtGSTzg5exJiZRfDoieH0L5LWGemqf+Bj+lStA9bo/pZSYg0Ok/+mHOLnzsi4Koe1biT38oNfOjwDXxTjXS+ZnL2FPJEFViT28B7W2BmmYBNatwcnlyb26l8g9d6I3N2GNjpH67g+89QG9pZmaRx7C37myGnGxU2lyr+2lfOIkuC5qJELtk4/ha29Fr69H6BrxX/lUlSgZZ8+R/NZ3q5F6ramBuk99AjUaRautoXj4KKnv/mBW1Mq3cgV1v/Qk5a5ThLZtQfH7yb/9LorfT2jndlAEmed/RnHfgUUpj7u5IuUjp/Gv6agKdOrN9fjXr6R0Cd0vvaUe/9oVs8hRpasHa/zywQDh0z0j6BlQfDr+VW0LPu/LQlWXLFXkVsxfWBumS+GyxEoI8RXgE8C4lHLr1LI48F2gE+gFflVKOSm8J+evgSeBEvB5KeUSVOKmcc/vb6Rpo9fmKr1jo/oUpCNxHYlQBIomsMo273+tm3M3iUBozz8d5tyPji3Iv68yUaT3x8fo/8lJLxUqvWJy17TnrS/ofeY4/c93YVcuTkzsosngC6cYfuXMtACf43o1WzNmUcX+DO9+8bnLEsBlfLhYvbuOnY83cXb/5DUlVkIo1ARbKJmTmPbFIy9CCIK+q6cfZg6MYQ6Oze5M3XuA0O2bPaPeKeitDYR2rqf43jGyz+2dbTArxPTA6Uov7SJA2g7CcXEL5VmRr/MI3bYBNR4j/fXnq6KT9sQkbtnEv6aDyJ7bqJw4O2tQFopC9vk3MM4OXpf6xDnQNCK37yKwdjWZF17CGhkBVUWPxxHBIG5peqLkX9VJ3S99HLN/gOyLL+MWS+iNDcQ+8iB1H3+c9A+ewSl5z5h/xQqyP3+ZSncPtU8+Rs0jD5Pb+yZC16n56EME1q6mMJFEjUap+8TjqLW1ZF9+FWtkDDUWJXr/PdR9/HGcXA6zfxCnWCT785cQfj91T34MvamJ5D9+pxpRc01z1v00+wcZ/9uvoDc0kPjsr1708s8Tr/SPniX20B6i999L+dgJUt/7IXVPPEpo62bKx0/Mug8LQeX4GeyH70BvTAAgAn6CW9ZSPnQSWZk/LR/Ysg4lMq1d5ZoW5QNdC4qYCVVB+K+wG+eyB7mCbaV7RbY4tyoWErH6B+C/A1+fsexPgZellH8hhPjTqZ//BHgCWDf15y7gb6b+XTKGDqfIj3kPv6IJVt3bhG04DB9OY+Qt9KBG6/Y4Vtlmsn9pWhzXA67h4C6iPko6Eqe8sOiJazoLIkLSlTiXIF/n17nWMgvLWDy69iYxig4Dx669dpvplOkn7jBjAAAgAElEQVSdeJeikbzoOkKo+NSrZ4CL8FJ8vpYGlFgYxaejNScAUZ0IgGdoK3w+Kid7Z5MqWNoLX1HwdTQjDRPjglZ6O53FGknia29ECfpnHc9JZzH7R28MUoXXQauEw0jTwjjXizU27kWhOOtFKKYGdeHTidyxC6Qk89Ofe+sBRm8fqAp1n3wS/9rVlI54ndhuuUzx8FGQLuE7diNUheKBQ6jRCJG770Sr8yKb/rWr8a3oIPPCixTe/aD6u5CWTeI3nia0fRtmv1ebZKfSXvqxXEHaNlYyeXH7G9fFLZWxs1mkffH3oVAVyl0nqXSdwtfagn9FB8XDR6icPI2xdRO+tjaErsMCNeDOwxpLYXT3oTXEq0Kb/vWdaPV1WINzJ/Ui4Ce4fb0nX4Pnc2iPTGCcW6A0kBBeG/QMSNedsvK5Og+brJhL50Y3yPN+o+GyxEpKuVcI0XnB4qeAh6b+/zXgNTxi9RTwdekJLr0rhKgVQrRIKZdc3XbqxWnPp+bNtbTtTPDKXx0hMzA9e65pDfHwF7cTrPHNWr6MZdyqGDiW+1BIlSsdRjPHqFg5XHkJaRDpUraukkCoqhK+ays1j9/rZaUKZaRtI4KBWaQKQAkFPOXlC0nVEnE+QiCNeWxTXNfzYZvqomLGMV3LuqE6o6Rle6mwrZup/+zTFA8cotLdgzk2PkvtWvj9+FZ0YI1PYI3PsNeREuNcHzgu/vY2L3UHSMPALZdQdB1ZLuEikKaJdBykbU2RFQis9uxTFJ+P4JZN1d2qsShCUfC1NM+OKF4DOBmv7sktVzyfwanUpzRMT49qCV2hslyhcrSb0K4tXlMFUxY3m1bPS6z09ib0jubpNKCUVI734C7UykXKOTII9liK1Fd/VK1Du1JI08ItLI+bVxNLrbFqmkGWRoGmqf+3ATO9NQanll2VtoH4qihCCLKDsx+C7EgJRREkVkUZOTZ5ka2XsYwbE4GIxqY9CTY/1ECswc/M+u980uT5/9pNeqiC5lf46BdWsXp3LQio5G1+/jdnGTw+3amm+RTu/fV2VmyL8exfdpMbn11vtumBeu5+uo03vzlA97tejYcvqLD+ngSbH6yntiVAMWNx7OUJTr2VolKwGckcv+w1SCR9E+9jXwWBUP/qVuK//jEqJ/vI/OgV7Mkc0rTwr+mg+U8+P2tdp1hG6CrKVfIIk7aDWywjgv5ZRrcAqCpqLIxrWLgXpn1utJm7lJS7TjL+1W8QvedOovffS+zBPRj9A2RffR2z1xN6VXw+FF3HKpbmkBxpWriGgRIKzvbrm0rPSonX6TEFUf0LlHAYJRgk9sD9yAtSXm65glMseQXoiywgXzCknK79kdI7TpWgyNknu0hUTp7DGp3Av6odmNKo2rae4tuHZnf5qQrBHRtQZ9RIucUypf1zdRgvehm2MyddLV0XeyyFM3lzOY38IuGKi9ellFIIsejXihDiC8AXFrNNJWsSbQrSdluC8VNZHMtF1RQa1tcQawlRzt2YhcbLWMbFoPkV9vxWB/c83c7Rl8c5uz9D584atj7SyFBXjn3PjlCc9J5r15acfCNJarDM1o80sHJ7DcHY7MHftlxy4wYb76/n6IvjHP75tFGsqgluf6qFeGuQyWFvAND9Co/+y9Xc9mQzQyfzjJwqUNcW4FN/toEDz4/w0v84Rzm/sEiMeZUEQrVELUo0TOnwKayRqfSjquBf1TarswrAGhrHLZQJ7dpIpevc7MFGVaZm/DNeT1J60S9/xNPuubDGSkoqJ3sJbl9PcOsarNHklIks+Dqa0dsaKe3vuv4edguB62L29ZMaGERLxAmsW0v03rup/+yvMv7lf8CeSOKaJq5lecKVF0SQhE9H8fu9FF2VpCzw0OUSTr5A6p9+iDU616xYWta1I1UwdZ7Xhu06uQLlgyfxdbZNpwNXt6O3NWGc7q2up0RCBLetn7ahkRKjZwBzcOHmzdI0cXKzAwmK34cSiywTqxsYSyVWY+dTfEKIFuD8kzIEdMxYr31q2RxIKf8W+FuAhRKzkWOTjJ3M8Oif7iR1Lo9RsPCFdOrXRBk/nb0lJReWcWsjmvCx/dEm+g5n+elf92CUbLr2ThBJ+PCHVPoOZTBK3gDkOpLeQ1l6D2eJ1Ol0bInN3aGE3kNZ0sNltnykgeOvTWCb3tcr0RFkxbYYR1+aIDsVyVq5s5bbn2rl3X8a5M1vDmCUHHS/wiNfWMVtTzZzdn+G469MzD3OfNcSaMaw85cscl8I7FQWN1ckfNc23HwJadn4OlsJ37F5DhGyxtIU3jxI5MHdJH7r45SPncEtllGiYbTaKIV3DmMNTZ+/NG3MgTECm1cTe/QujDMDCE3FzuQxz3qvqvLRMwR3rCf60bsQmoY5NI5aFyVy705kuUJh7xX143x40FQvnem62BNJCqk0OA7xz3wKLR7HnkgiKwZmXz+BdWvR6hPVjj6EwL+iAzQVc3gE6SwuzWmc6yO0fRtaIk75VPdsEqWqc6NjTHmcqmo1nXjDwnYoH+sm+ug9qFGvrlAJBQluXz+LWPlWtKI1TnvbStOicqwbuYgUnls2sFMZpJTVdKISDKA11GH1Laud36hYKrF6Fvgd4C+m/n1mxvI/EkJ8B69oPXsl9VUXopwx2ful42x4tI2WLXUE6/xYZZsjP+7j1EuDlNI3wSxyGcuYAd2vEIxqZMYqVAre4FXK2WTHDFZsi6EHVMheMKjJS5em5JMGp99Os+2jjcTbQ4yfLYKAzttq8Qc1Tr+dwqq4CAEb70/g2C4n30xhGg5C8aJep99Ocfdn2ujcWbMgYiVQaK3bxkS+m/QV1muYfSNkX3ib6IO7SPz2J3ANE3s8Te6Fd4g9ed/s9m7bIfvTt3DLBqFdG6n55AMghKdVNDzBhekeadkU3z6M3pwgcv9tRO7b6RnavnmwSqycTJ7J779M7GN3E75/J1G/jrQdrJEk2WdfwxwYnbHDKUPZG6wzSvh9RO68A7dUwkqmkJaFGg7jX7MKt1TGLXg1PtKyKHxwgMCG9dQ+9lEKH+zHLZbQGuqJPbQH42wvlTM9iw7+VE6fwewfIHr/vQhVwxgYBClRI2G0hnrKJ07O0qfCdbHGxwnt3E54104q3T3V87NGp2uXhKYhdA01GkVoGorPhxqN4AiBtO1Z9WPXEtbAKNbAKOrmNd4CVcG/cTVKOOilAzUV//qVs6QSnHTW065aDGwHa2jMU0ufSk2LoB/fylbP3+/COsBl3BBYiNzCt/EK1euFEIPAf8QjVN8TQvw+0Aec73v9CZ7Uwhk8uYXfvdonXExWOPDtHhRdoGqK5+tmLetoLOPmRKVgMzlcoXV9lLrWAJnRCvUrQrSsizA5XFlwGm4mHFty4rUJ7vx0K6t31zLRW0TzKWx7pJGxs0X6zxe9C2hcFSIa9/H0f9qEM+N7pAdU/GGNSJ0PTVNxnEuPrIpQ8esRBEsTGpwJaVrkfv4Opf1dKEE/0nFwsgXcQhlzYBTnAm0oN18i+/wbFN48iBIJIRQFadk4+SJuYW7XlzWSJPnVZ9HiMYSmIi1v/zNhjyaZ/PYLqPEalIDP218mP6dIXhom6a//8xVf81WHUPC1tRLYsM77+XxXXsUg++IrmDPIitHbx+Qz/0zNIw+T+LXPVJebg4NkfvJzrwhcXZzwq5PLkX72eWoefoDonnuIabpHeB0HZzJD5VT37A1cl9LhowTWrSX28INE99wHtk3pxEkmf/Qs4EVq4r/8FHpbK4quo8br0OrqaPzC7yEti0r3GTL//MISbtbiIQ2T0oET+DetrqYDfR1NaC0NmGf6UcMhAlOfgZcGrJw8iz2x+Bpgo7sPN1dAqfc6LoWiENy6juLefUva3zKuPRbSFfjZi3z0yDzrSuDfXOlJXQ6BGh+R+gAImOwvIKaMhZ1lgrWMmwz5lMlr/9DHJ7+4jj/4m9vIJw2CMZ1K3uKVvz+HWVrajHSwK8/I6QJbHm7gyIvj1HcEad8c5ZUv986uRRRQztsceXG8mnKciYkeg02tTxDQLi8QGvbXXx2BUPAsbOYRUKzWXF0IV+JM5nEmF2ZMK0sVrMt0EkrLxh5LXWZH8uLndB0hKxXSP36uqnouFIFrWTiZKXX0mRG2KVJTOduLFq/z7E3KFayJiWmTascht/ctFL/f6440TDI//Xl1P06hSOr7P8atTN9Te2yc1Pd/jBavQ41GEELxFN0zmXm70OxUmuTXv4XWUI/i83lK7dlpRXPXtMi98fYc5fDzcIpFpONg9g8w/tWvY4545LF8ogtrZBQ77ZGQ/DvvUzx8DKdwZfI8laPdOJNZtLin36YEAwS3rsU804++ssVT+D9/7sUy5cOn59VNuxys4QkqJ3oI79ldJWp6RzPh+3eRffa1a1urtowl4YZXXp8JocCK2xu463c3UNsRxjYcfvjv3kG6ktueXs2+b52hMH51WlBvdaiqDyldXPfGaRH/RYR0oZAyyU0YnN2XYexskVLWYuR0nlxy6fphZsnh6MvjPPy7K+nYHKV1YxTHlpx+Nz1tOC1hvKdI24YoJ15PMnRibjGsqvjY1ZkgXx7FsC8+EAkEmhq46OfL+PAhKxWs4YVXYrj5PGb+4sTUninH4Lqz923bmAPzaDPZNvb4xOxtL3UO5TJm/8D8H06Rpsvuo1TC6Jm2WnCyuVlWO3YyBVyGMC8A1kQa41Qv6l3bq5Ywgc1ryf3sbQIbOlHC052q1uAoRk//ko4jKwbFtw8R2LoOLe6J4wpdI/rwndjjaYrvHl6SH6UIeCS5Sp6XcdVwUxGrmtYw9/zhRtJ9BfreH2fzEysQisAq2bRsjdOwrmaZWC0IgrbWuykUR0inuy+/+jKuGYSAtXfHiTX4OfNBmrEebybvOhI9oGCV574whQLKlOCgogiEwjRZmoEz76XZ87kVbHqgnqa1EfqPZkn2T6fRpIQTe5Ps+mQLd366hddzFoW0R+Y0n0KswU9+3MW0SwykD15SIFRBJeRPXMmtuKGh6gFvInKdDMo/TPjiYRruX8PkoUFK/UtvCIqsaaBu9wrSH/RSPHflRGYpqL9vDdG1DdXOvPHXTlPsvUrnYjuUDp4kuHNTVdNKa4zjX9OOb3UH4rwoqOtSPnx63rT0QmH09FN8+yCxJ/YgVBUhBEpNlNrPfAwlHKT0/lGcfPHSBEtVED4fak0E36p2gtvWUXzzwOLrvpZxWdxUxKpxg8fW937pOKE6H+sf8fySKjkLx3YJx+cPES9jNjTVT0P9ZiyriFfce2MV3t4QUMRlDVKvBiQwOVQmGNX4tf+yBaviICUYJZuBYzne+EY/yX7vhdywMkTnrloCYZXVt9fhD2ts/1gjDZ2e1Ej3u2lyE9MDf2qgTP/RLJsfbMAfUXnmL07jWLOvqf9Ilje/OcDdT7fRsbWGiV6P2NU0BdB0hR/9n6cZHjpMxcoh52NvU3CRFI00jnvrEQ/NF6Zz+ycwSxkGT72C69zaM/zo+kZWff4elO/suzJitbqeFZ/ZhTGRv27ESov4CbbWEmytIbqhiWJv6uoRK8A40481OlGVXlBjEQJb1uFra6qSOSdboNLVc0UpO2lY5F98B9/KVgJb11XrurRELbWf+RjBXZupHOvGGhzFyRa8KJSiIHQNJeBHqYmgN8TRV7bia29CrYkign7KB7uu1q24tlCUKfV6gdAUhK4jfNN/1GgYJTg7Yq4EA/hWteOks7imhbQspGl798ZxkHKqC+gaeB3eVMRK86tYFQcjbxGqm9azEYpACHA/hIHww4XA748R8NegKPospWDHrlAojFaVroVQCPhr8fu9FnzDyFExMrMGQ00L4PfXEIu2EwzECYcaidetmZJ8keQLQ9j2csRP6Drhe2+ntP/INVck7tgSY8fjTfQfzTFwPIdluCgK1LUG2fbRRgIRje/++QlcR1LfGWLn403VaFX/kSzx1iDx1iCW4TB2tjiLWFmGy4HnR/GHVMp5m559cwtdbVOy9+v99B3OsumBeho6QyiKYLS7wJkPJpnoK2CUsnO2uxASyUBqH85VEAi90aBqOqFYM4qiIRQVbnFiVR7KMPHGGXInrlpD93XD6M9OMPpiF4171rLxi49e9f2f7/TzrWjxpCI0lfC9O1Gi02lA89zgvKrsiz7WZI7Jbz5P/Heewr+hs5p+VAJ+gpvXENi0GlkxcMuGR+KE4kWppsiV0BbXgHDDQFGIffwBAutXIgIBhE/zonaaCprq/V/X5phV6y0NJH7/l5GW7YnU2q7nEOA4SNNCVgys0STZZ1/FzV3d9/xNRawm+wuE6vysureRcsZECNCDGp13N+IL66R6Fla4enNA0Ny4g/b2e1AUDST4/TE0LUClkiE12c25cy/h2ja6Hqaj/V4aEpu9dQHXtRlPHmdw6J2pyBTUJzbR3LSTQKAOXQ/S1LidRGK9dzgpOXHyB+QL88qO/UJBa6ontGsrlWMnrymx0vwKD/zWCoJRjW//2XEmR6ZJre5XsE2XdXfFCdXqFFImXa8n6Xp9cYXSC9nGNl16Ppik54Mr6zCynKWnOm5kGOUcZ/Z9F8cxcaxbf+JRGpjk1H97+XqfxtWDK3Et59oE5l2X8qGThO/ZWa1/0uqm9eWk41J67+hVq2OyhsdJfvn71Dyxh9Bd21EjoeqEWwiBCAbmRG7mhZRIV86WLrlRoQgCGzoJbt+wqM2EpqLGIpdcR2uoI/+zt3D5BSZW46eznH5liAf/520YBYvajggf+V+3Eajxcfz5fpI9t44SbTBQx8qVD1EojNDb/xqmWSARX8+6NU8yOnaI/oE3cFwTIRQ62u+juXEHI6P7SaZPg5Qk4utoadmNEArnel9BSptksovJzFli0Q42rv8UA4NvMT5xDDn1xjHN62BirSj413YSWL8GJRz0xAGlxJnMUnj7A5xMDrWuhvAdt1E6dBTfyg78qzrAcSm8dxBrcEokTwj09hZCOzajRsJYqUnKR05gj02TCrU2RmDzenztLQhVxZpIUT7aVV1HrYkSvG0rwQ1r8bW1UPNLjyErFVzDJP/KmziZq/t86X6FaL2PUs7GKM1uItADKuFaHbPsYBs31stPVXz4tBCKULlQJ6piZT/UqJUQCr5gDa5jYRlzn1/NF0bV/ZjlLNKdTsWoehBND3pehLaJZZbmFKopqo4vEAMhcF17KgV4qdS5QPMFUTW/1wHnOjh2BccyLrHNtYHiU2l6ZCNW3jt27dY2sidGyBwepP6+NUQ6E6T29TF5oB9pe9fd+PB66m5bUa0NGv35CTJH5p9oCV0lvDJO3c4Ogi01SCRmukT2+AjZo0OzBmxFV4nf2Undzg7UgEZ5NEfqvXNemlHO2N+KOLU72gm21iBUBTNdZPLwILkTI9VzPA9/Y5TEHSsJdyZACApnJ0i924uZ/vA976zBMcyzAx6hmpFVkFJij6eodPdd1eM5E5NMfu9nlI91E7l/N/51K6dkRsQl/Q+llGDbuMUy1miK0sEujJ7LNwMsY/G4qYiVY7oc+HYP6d4C7bvqmewvYBQsBg8kOff22C0lt+D31xDwx+jrf51i0Qsjp9PdVNozBIJ11S9QwF9LY8NWJjNnGRh6B9v2ogalcpJAwPtsdOwgpdIEtlPBdiqYgTokEssuUzGyXM8aK9/KNup++Ukqp89ijycJbt2I3tzI5I9fqPqxKaEQ4Tt2oDUmEIqCUyyihkIo5+1NBPhXr6T2009gjU3gZPME1nQS3LiO9Ld+iJ2anDpWB8HN67FTk0jbIbx7O4E1naS++UMvMjUlLOlMWXjYqTRuseSFja+B8KBRcqqSCHd8qo1zByZxHUm03s+6u+OsuaOOt78zWBUOvRx0v4IvpCJdSTlvz1vQfqXwaWFWNdxLIrIKRVFRhIYrbRShYTlljg/+M7ny6OV3dJWgagFW7/w0plGg98hzOFZ5xmd+OjY/SijWzOn3v4lVySMUjZqGNTSuvJ1gtAFQMCtZ0kPHSA4expmRCg/VtNC57RNovhCq5ic9coK+o8/jOnPryFTNT6J9B/GWzfhDdQhFxXVMCpODDJ56BbOU+TBuRxVCU6dISi3GRJ5gWy31960h9d45gq21+OvD1O5opytVpNDjdezZBQMrVya8Mk789k5yJ0fnJVaKX6P50U20f3oniq5SmcgjEOi3BQk0xTwiNEWshKrQcP9afIkIZrqIGtCov28NDfevoesvfkZ52Esz+xNh1v6LPeg1QYyJPAhB3W0dtDy2mdP//TVS752rvqZCK+Ks+zcP4q+PUBn1JjuJOztJ3N5Jz5ffrO7zw4JbqpB/6R3MoblWNfZYCid99X/3smJQPngS43Qvvs52/Os70dsa0RI1KJEwwqeBK700mGHiZHLYqQzW8MT/z957R9lx3Xeen4ovp84RjZxBEiSYRIqUTCpRokTZslaWR8f22B57PDtr74y9nj07Hu/YM2dtn53xOM2xx3PWtmQrWYkSLVqkxJxAECRy6AY65/Dye5Xr7h/18BqN7ga6gQbApvE9B3iv36t369atqlvf+wvfH/bQBM7ENH65uqr4ImE7VF57B+vcPFH0sgX8613eyfepvH4Ua2DtvSl+qbJmBdwvxroiVgCO4dH3/Dj9r0zWBEJ9fMd/z4Vf+yIIrpNljQurZOnCg8xzuTDLhMNpQnqCYmm0TqoAPM+iWBqjuWkP0UgT1erK0p1vNMI7t+GbFsVnXsAvV3Empsl89nGc0XHERZo4ciKGcBwK//g8XqkSaO3UVIelUIj4w/fjzs6R+/vvISwLtbWZpn/+U0Tv3Efx2ZcAME/3YZ0fwK8agETsnjtIffxRlHQSv1zByxepvPYWvmES2thF5dVDdVJ2PeC7gle/OooeUbj70+3c/9kgGcP3BEbR5fWvj/La15dIYV8G3XsS3PVYC/EGnW///jnyU2s/4TXEN9KU2Mx47ji+cGlObmM8d5xMbAOOZ1CxbmxZKdepUi1N0dixj0i8mXJuPqU9FGsg1bKN0uwAjhVYMtKt2+jZ+wlss8DUwJv4vkuqaTNdux5B0cKMn3uprs1ULUxy/u1vEkk007P3MVQ9srRFQJJo23w/HdsfplqYZG78BJ5dRY+mkWUV3705MVmSJBHKRDn3319ACevs+ncfIbW7nTP/77OEmhPs+LUfI76lqU6ssoeGyL41TMOBDaT2dCzbbmp3Oz2fO4AxWWTgb17HGAuIjJoIBW43e34hIGsK0Z4G+v7kBUq9U0iqQtuHdrLxC/fRcPdGxp4MdM/suQpDX3sLa7qEUwzuz/jWZnb9+odo+9Ausm8NIVwfWVfZ8L8cINycoPdPX6j3PbO/m23/6wdo/+geBv/2IL59A7WdhMA81Y95qv/G7bO2X79sYJ7owzx1LoijCoeQNLVWJ5NATsHzEFYQUyScq18gCtuh8vLhtev/SuELKq++c+P3ew1YV8RK1mTadqdp35NBj2l1k/UFnH9pgqnTN3ZleL1gVGcpFEfoaD+AED6WVaC5aTeKojOXPYtXWzUrSgiBwPMWP0Rd10QIH02NLPruXQPfr4ne1c7lxe8vgrBdzJO9eIUgju5ioT0lmSDU00n1nRPoPbWK85qKb5jomzZc1IaNFNLROtuRdR0ppNcDH28WZoeqfOf3zpJqDRNJqEgSOKZPOWdTztqrsjoNHi1ilFw+9W+3oGjLuwSuBTG9gUJ1nMHZN0iEW0hGOpgunGWuNMDurscIawkq1o3NAMuOn6Spez/p1u2UcyNcWHSkW7ahqiHmxo6D8FH1GG2b34fnWvS/823McvBQnhs7Ts/ej9G25X3kJk5h1D73PRujNBW48y6T1BFJtNC25QEK0+cYPPY9bHPeZSzJygIX5A2FBNZcmXL/LGoijDVTwpwqUh6cw3c8PNNBS14yNwgRuN2W93bS/NA2ZF1l8G8PUjg+X69uKTec8HzmXu8ne3io3ubsGwN0Pn4bsZ75Onq+45E7vFDnqXB8jPLgHOG2JJIsIYBod4aGuzYw/v0T5I+OIGoVAWZf76fz8X003rORse8exZq98S7BmwpfBJaX62B9uYXVY10Rq+79jXz4/9rPbH8xqAt4yc2vRdZp1sMScNwqQyMvsn3r43R3vQ/PszCMLGd6v00uPy9+57pGIM64BHnS1AiSJOO41UXfvVtgHDtNdP9eGj7/abxsHq2rHePYaZyZhQ9n4bn49tKp/JKuIakqkdt3E9q2acF3F9SWUVVi9+4nduB2hO3gGwZyNFKvv3Uz4Zg+s0MrO0dqSCbZqOM6PvGMRrXgUpi2ECKwdNlVb1F2rBaSSbXohGIqVsUlN2EhgIaOMPlJC9cO6gamWkI4tk8lt7yFRSDq1lTPd1FlDVlWcbwqEhKhm0CsqsVJKrkR0q3bmR48hG0WUPUY6dYdVItTlPOB1S8UTRFLdTA7ehSzPB975zkG+amzNHfvJ5burBOrlUEinu5E1aPMDL+9gFQBN49U1eBZbmC9EQLP9nArduAiqv27dHF6JShhjUh7Cmu2jDF65UWs73qB1MJFl6RnOPiOh6xf9PiRJaKdadJ3dBHb2IiWCKOENRJbmrFmy/OhD60JtGSY5ge3ktzZxsUNx3oa8R0PJarDGgcjrzdImoaWasQpZBHOlSVQlGgMWQvhFG6sxfm9inVFrLSoSmGsyg9+9x2ssrOIWPnrIcNhxZBobtyNaeY4d/5pTKsILFZKN8wchpkjldzA1PSxegagqkZIJjfgOFWq1YVZYUJ4wQpevflK2W6ugDM1g181sMcnMU71YvUPraq4qF+p4lWqVF4/TPnVQ1x8YYgaydCaG0l+8AEqbx2h9OIb4LqEakHqS+PygaA3C42dYX78321l9FSZSFIlmlJ54UujDB5ZPrB+w74E+z/SgvAFmfYwB78zyfnDeR79590cfnqavoN5ommNj/7KRt5+eoreN5Z/YFatLKlIO7oawfGqIF73V8sAACAASURBVEm0p/di2HlCWrxuSb2R8ByT3ORZunY+Qryhi+x4gViqnUiihYlzr+BaAWlVtGCh4doVLp08XNvA9120cJxVabtJEqoeqwXPv8se5oKFWmxCBAHM1wJJQlIkhOuvrC0B3qVuuUt/Jks03buJTT93P77lUjgxTqlvBuF6hNuSCzaVZAlJlrDnyhgTC+NDjYkCbsnCLb/3tNSWgxwK49vWoiLgocYW2j/xU0w+/Q2MscErNKKQvuNewu0bmPjeV4L2bgASkVbi4VbylWFMpxQ8l665zTZUJdCz9DybsjmNvwbtrhbrilhNHM+Sf6CV+39+B7mRSi1Yff6CGn17jrmB947kgqZFiUQaaW+7E8cxEAhc16BYGqVanam5CItMTr1DZ8e99Gx4mLm5swiCrMB0eiOTk+9gmAvjhGyrhO2UaW7ajWFkcZwKsqzdFB0rNZNE72gl/+QzWAPDwQQhREBqVvgQ8MsVjOOnid6xB3tkDHcmi6SpqM2NOFOzeNlcXc9FWHagWpxJEdm9HSm0WFRWVA0kTUXrasc3zCArrGqAf/OJuyRDOK5y9EczzA4bvO8zHdz1WOtlidXYmTK5cQshBLc92sy2e9L0Hcxx/nCBvQ83cu5QnvYtUfSIzOT5y1vOcpURVCWML3xcz2KmeI4NTQdQJI18dXRpa9WKn+VXT2SLs/249v1k2naRmzhDum0nnmtTmD5X74DvOQgEsqIv+r2sqEiSjOfaq+kwIBC+iyTLyMq6mk6vCr7t4hQMwjtSaKkI9txKyOTlx1ONhWh/bC+yrnL6D56hOpILLFqaQuN9m9AS8wtAu2DiVm1yR0YY+97xxXIBAvxriCNaT1DCUTL3PETu8Kt4lUuee5KMrOmBsOaVIHyM0UGcYv6GxQNKkkJzcgc9LfdhWFlmin1M5E5QvUZrd2fDHbSl9wBguWVODD9J6QYm01zAupoJ4s0R2nZnUHSZxi1JfG/hDVucNN4zxEpVw9hOBVUJ0dS0GyH8mssvjGWXON//A3L584BgfOIQAK3Nt9HcuBtqMVcTk4cZG39z0UrAtAqMjr1Od9eD7Nj2SXzfxXENTp/5xg0nVsL1QFbIfO5TCMsCX+DMZik9/ypW38A8uRIsOz8Lx6X80hvI0SiZzzyOpMqBq8OyyX37abxsDncuR/XISeIP3EP0jr0Ix8EeGcfLLbbO2KMTWH0DpD/1UYRl4eUKZL/+vSW3vagXK3seS/X/rhrVgsPMkIFRdJkaqNC9J4GsSvju4g5IMnTsiLPnoUZkRaKxK4xZDq6HvkN5bnukicauCFvvyTDRV6E0d/nVvuWWGM2+w4WDHc8do2hMoio6ZXMG178xq91F/armKM4OkGrZQjTVRrJpE+XsEGZl3lrrmCVss0gk0YyihvDcWl8lmUiiBSR5gYtwRRACozyHJMnEM12U5gYvq1C/3iFcn/zRMTJ3bqDtkZ0MTh7Eq9YexrKEElLxzMXehMtBVmX0TBS3aGJMFPAtF6RAuT3a3YCTnyf71eE5KkNzNN63mZmXzwVWq9q+5LCKrCsLguffy9CbWolu2Ezh6EGuySYjBNXhGxt4r8oh0rFuNCWMFu0grKcpVMevmVgVqmN0N90dqNIrIRrim24Rqysh3RXDMT2+8+sHMQv2IlO0Z783JjRJUti88RES8Q5Onv4applHIJCQiEab2Lbl47S13kGhOITvu7iuyfDIK0xNH0PXA0E0265g28uVIRGMTxwmmzuPrsUAcD0bw7ixsTFSJEzqsUcwTp7FPHMOYTtIqkL0wO2kP/lhZv78S/jlCu7MLLN/9VXcmeX75xVK5L75FGpTQyCQ5/l4tUw/AGGYFJ76IZU33kYK6/jlKl6hSOXgO7iXECa/UiX79e+iNTeCqiJME694ZcK+soLW0jVbNrSwQjgWqKlHkhqu7deDeC9FLKPx8E93cexHM5x+JcsdH26mZ1/gXilMWYyerXD34610bIvx7P8cXmGw/Py+fOFSNMYvsy01McIrT/2SdPVj43sOuakzNHbuo6nrDkKRFOO9L8yTJ8A2CsyNHqV9ywO0brqPubFjCOETz3TRsvEeijP9VPKXHIskIysaSDKSFFil/EusWuXcCKW5Qdq3PIjn2hRn+/E9F0XV0cJJKvmxmvvx3QslohHf0oIa00juakfWFZLbW7Hur+BVLYzxYiCDAEy/1Ediewsdj+0lsbOV0tlpJFki3JbEKRic+4uX8aord8e5hk3hxDjtj+1hy88/QPn8DOG2FMndbbUMwXk4BZPBLx5k+6/+GHv/4+MUTo7jFEz0VJjYpiamX+xl7MljCM9HjenEtzSjRHRSezuRVJnUnnY808Gt2Bjj+RVa3G4kJJoSm2mMbwZgLHuEsrUw5k/LNJHZfz/RDZvRm1pp/+TnEa6DZ1SZfu4p3GL+QlOEmttI7rqNUFMbTjFH/sibgWuw9uxMbN9Lct8BlFAEa2aC6eeeQnjz81iks4fUbfdQOnOU5O79qKkMTm6W3FuvYs3MK/Prja1k9t+Hlm5ADoWRFAXPNMi9/RqV82cWuyq1OLHwfG1Rw86vCQEqGZPYboWQFkeSFDLxDQzPHERwY7nBuiJW0715ihNV9j6+gbmBEq7tLzhhM30FihPrX/1Z1+NkMluZmj5KoTi8gBx5vo1lFVCUEJKkABduAoFlFbCslWq4CEwzh2lePzmBK0FtakDv6SL75W9hnRusfaigtbWgd7XXY5yE7eCMXOHhDeB6uJPLBx4Lx8GZWFhawplcrD0DgU6MvZJ9Xthe+DUZjMtDkgIRyWvJFoulVB76fCeVgsumO5K8+eQUkgzduxN0706Qagmx68EGhk+UmB0xKOdsuncnCMdUNt+ZwqkJjgoBx380w+f/007GzpaZHlhZAL2uRGlKbiUd7cLxDPqnX0WRVMJ6kpIxVRecvQAhFscGLgVJklH0KMtWlb4CytlhqsUJWjbdg1Gapjg7uKgfU/1vIMkqrZvvo2XjAYTvo2ghKvlxhk/+Yz37T1FDdO16lGiyFUWLEIk1EQqn2HHvF/Bdm+LcION9LyF8F88xGDj6XXr2PkbXrkcRvofwPSRZwTYKnH/7GzecWAnfx5wq4lYCYilcH2O8gDUdkCPf8agOZ7Fr1qBwW5Kt//IhZC1wHRkTBRI7W0nsbEW4PuPfP8H4U8cB6uQpd2SUxvs2kb6tE+H5mNMl8kdH6xYjp2RSGZzDLS+0YgrXozKcxbzQF9Nl+OuHcUomDQd6yNy3g/LALENffhM1FqL5wS0L5vncOyOc/J1/oOWDO0jubEXdEcItWRROjJM/Moqoueyj3Q1s+5WHazXmoDqaI317F+nbu/Btj7HvHGHyh2euy/hfLWRJpjG2kY1N9wCQLQ8uIla+WaXUdwLhuSjRGLm3Aleg8Dw8Y/46U8JRUvsOUDpzlOpQP8m9d9L6oScYf/JvsbNBm9XRQZxyicz++wi1tC+KK5XDEZK770BLpin1Hscf6CV1+920fuhTjH7ri/hmFSUSo+3DT2BlZ5l780VCDS00Pfghyn2nMEaHlgzpiIYaUOX5MIxCZRTHvfZnt+mU6vGekiQR0dPoahTLvbHi1+uKWKU6YqQ6omQ2xNl4X0utxt38929+se89Qax838P3HKKRRnQtXs/qk2WNhsw2IpEmJqeP3JRA4bWEl83jTs+S/PDD2Ht2gOejxGNoGzoov364pje1PiB8H89eWX+1cAJFC+NeZbBzfsri7Bt5Us06L35plP4jRSRJIpbWEL7g4Lcn8D1BPKMxcqrE8389wsbbk/ge/OivhgnHVJyadTc3YVEtuvQdzGNWr0z0dDXG1tYPkI521pX/B2cOEtZTbGv7ACdHv4/pLCT3vufU1MevjFA0jaJoCyxNK4XrmIye+RGxdCfV4hTOEmTGdQzGe18gP3WWaLINSZKxqjnK+dEF50P4HpXcGLYRHMscRxe0Y1ayC8ifUZrm3OGvE0t3EopmkGUF1zGoFqcwyze+ALFvugz+3Zt1r7NTMun77y/W4wStmRKn/+AZ/JqieWUoy5H/45vLt3dJALpbtph6/iwzr5xDUgIyJjw/KB1TC5jPvjVE/thY4Nq7CHa+yunffwYusk7a2QrDX3uL0e8cpfFnfxJn1iB/bBx8n+yhwUX7L/fPUhnOImtKvWC67/oIZ367Ut8Ub//bbyx9QCIgl+82SMgo8uIYwIvhGVWMkQG0ZAbftjDGhnCLSy2QBaWzx8m99QrC83DLBdo/8Tn0ptY6sfKqZTyjirN5B1q6YYk2AAnyRw9SOnsChI9nm7R96Am0VBrLrKKlMqiJFDMv/wBjdBBreoL49j1IqopvLr1YC2kJZDnI4hdCUDIm1yTI3PUsLKeEEAJJktDUKKoSvkWsLofRI3N89zffnP/gkuQds/TeKI7qOBUmp4/Q1Xk/O7Z9kqqZDVLZQynisVZK5XGmpo9yMxXT1wJ+pUruG08R3rMDNZMCVcXN5qgcPoY9OHJN1eBvNITvLllSZSmEYg1oofhVEyvfh4EjBazKwvE5/fLSqdKzIyazI4tj5xRNomlDBNvw6H+nsKLLqTG+kWgow6mxp1Fkja1tDwPBSlGVw4S1xJLEyrVXZg2LJFtQ9MhVESuET3F2gOLswGU38z2HcnaYcnZ4+W18l9nRI6vavWtXKUz3reo31xMLCI0A33QW/O0ZF/3tC7zKKhdqvlhEmi6GcP1aMsClX4CQVGL3HaD84uv1OnrC9YN6jC4IXwpCPTwfz1jaehm0v7xlU3hXcUw3GZIko8hrIwHjWRbW9HhQgJiAkPmug6ytrn2vWsWamawvJLxKGeH7yGrQjm9bCN9DTaaRNB01GkcJRxdYzy6FpkRqHhfwfQfTKbE2zzOB5ZRrbUmocmhJoiohI8lKbd2xhG6i8K6J6K0rYuWaHmXTI9EaoWlrklBcwy47zJ4vUpxcP9aNK0MwNv4mhpElk9lMOJTigqtveuY4+fwAtnMT6vpdB7izWcovvn6zu3HN8D13xWVLtHCCeOMGjOK1V7y/WoRiCvf9eDub96c48fwc+cmVEZmo3kDVmiNfHSUVaefCZOj5NkjUi4BfDM+1cMyV1VkMxRuJptpueAmYW7ixUFubiOzbQfnlgze7K+8qSJK0ZsQK38d3LjE2CFht8ozwvcuW9LLzWYqnj9H0wIdIbN+LEgrjlouU+04t+xtF1pAILJ2ecJcUuL5auL5Zi0kGWVaQpIWZkSEtQXvDPhLRtiCkBmnRkGSLgwxNvXbVfVhXxEqSYfODbTzwS7sIJTSEJ5AUCbvi8vpfnqHvhYm6btF6h+87zM6dZnbu9M3uyi2sAMJ3MUoz+J57xQBsSVZo2ngn2dHjeM7qsjDnRk2+94fnsY1rs+a5ls+5Q3n6DuaYGqguyrBdDo5vEVOaUC+Z/MNaEgkJ11t8PL5rY5Zm63FHl4OqR2jcsJ/idP+SNflu4eZCacqQ/NBDVF5/m/Ce7YS29OCXKxSefh53YhoUhcjtu4jcsQclEcM3LMxTvVTfPIKwHZSGNPEH7ya0cyt6ZxtNv/TPwPNwJmcoPv183f0vR8MkHr6f0PbNgKD69gmMd04iaiLBSiZF7L470TdvCErKnOylcvAdRK1undbdTux9B6i8fpjYvfvR2lvxcgUK330Wr7C2xdTXEitxBdZRk6WRbrbenu/jVUpY0+PkjxzEtwzsfBbfXN7YsaDPQiyKy7y27ngXDFaBZeoiYiVJCps7HqY1vYtidRzbrS4ZA+ZdY3bzuiJWmQ0J7vv5nZz+wSjnX57ErjjoUZUt72/nnp/bTnaoxOz594bcwi2sPxiFSRyrTCiavuK2iaaNNPXsZ/r8wVWl5zumz+S5a1fS91zB2JnVWz2z5QFakzvY3v4oVTuHKodoSW6jKbGVipVdVnW9kh3Fcy1UPXqFPUikO3aR6dzD3PAR1ru7+70GWdcJ796OkklhD45Qee0t5Fi07s6TZBmtvRW7fxgvV0Btayb5sQ/il6sYR07iGybGyV6EECjJBKXnXkVYNr5p4lvzD7Pwzq34VZPK62+hdXWQevxRhO1gvHMCORoh/ZOfANelevAdUFXiD92LkkpQ+P5z4HrIkQiR23ejJOLYA8PYg6PIkfCCjLd3I1bjCnTLRWRVJ751F9Xh84CEnZtZecF4SUKJxpF1HSUSRVI1tHQDvmXiGZWVtyNLRHu24FsmvmUiPA81lsAVAt9aeuHo+Rfq3UrIsrpyMrkCqEqoboESwl8wv0pIRPUME9nj9I4+uyaipEv24bq0ep3QvC2Ja7gc/cYAdnX+pJemB9j0vlaatqVuEatbuGkwS7OYpdkrEitJkpAUjfadH8Cq5ilM9t700icrRcWc4/z0y2xovJuG+EZUWaen+V6K1QkGZl7H85eOc6zmx7GNwhWJ1YWsyc49j2BXc5Tmls4quoWbB0lVsM4PUX7u1aCoryzXg+KF41D64ctImgaKjDszR3j3NrTOVowjJxGGiX1+CCWdRFgWVt9A3coUNB48EZ2pWUo/fBkvV8A6N0R4+ya0zjaMd06gb+xC72oj++UncacD3TElnSR2737KrxzCywZuZElTqR4+jnHsFHj+gn5eDrKkIMtaUAbIt9fUmnIlBMRqZSTDnBwlf/Qgyb13ktx3ACc7y/Rz38MtFxGeW3+9AOF7eJUSfq3EjRKO0vTAI+jpJtREEknTaX30U7iVErm3XsGcGEG4tXYuIidBIHyxHrsl6yHcapnE9r1EujcFMXS+hzk5xsyLTy8WLyUo2eYLD6VGJMNa4lqG7SJIhPXUvJvRt/HF/JzkC4+J7DGa0ztIx7sxrNySsVS+7y5pfV8p1hWxUjQZzxW41sKB8CwPz/VRtBWozN7CLVwneI5JYfIsyZbNi/z6SyEUa6DnjseZOPMi2dHjuLbBtVtoLpTiEdeFkAh8suUhSsY0ET2FKoewvSqmXcT1l5+IbLNEaWaASLLtiq4LSZKIJFvZeOcTjJ78IcWpvqsLZl/c8kWK/rfI2tXCtyycscmAVMECsiKFdKJ33UZ45xZQgow9raUJu3/5RIGl4E7N4JeD4OfAomUHxdJlCbWlCSWTJv3ER+rK61JIQ9gOckivi2X6FSOQV7mgzr4CUhXR03RmbiMeasHzbebKA0wVzwYxhBdvp6VoTGxe1TGtBJoSIaTGVrStb1tk33yRwonDSLKM7zh41cAKbc/NMPbk3+FV50mNU8wz8f2v161InmUw98YLSMol7nkhcCtBO8bYEOPf/TJued59as9NM/69r+BVykiKQmrfAbREmvHv/F0QsC5J6OlGWj/8BOW2LsrnF4ezGE4Bz3eCWCtJJhXrYqpwetmF2UoRUmPEQo31Ocb2qrgL4rcExeoEnU13sm/Tj+N4Rk0KZuF8MFs4x/nxF666H+uKWOVHK0QzOhvubmb48Ay+I5A1iQ0HmolmQuSH1yagW5JkJEVFllVkRQveKyqyrKFHMyuK/ZMkmUiqHSEEvu8GRYQvvHpOcDLX5Uo8EHGU5NqY1N9rhBPNV4yhCVoAPZomlunC9x1875Lx8d11Y8G5GEL45MdP07btAfRI6oq1BiVJIpJoZsP+x8l07WV28G0q2RFso4DvXX6CkWQFWdVRFB1FC6FoYRQtghZOoEeSVPPj5Ceul0aPwPGqOMbKXZK+a5MbO0VD1221enyXhyRJxDKdbL77MxQme8mOHq/LH1zp2pBkFUXVg/HRwihqCFWPoEVSaKEYxZl+ypdoXN1MLJhr6veVRiiWWbFgqhaKE2voDsr21O4h33MRfjDfrOn95IugYsKiA5EI79tJ8mMfoPDkM9jD46DIZH7yE6vehXC9SwSgL3rv+3i5PLmvfw//YmuX6y2sjuD786RqBVBkjU3N99OVuaO+MGpKbEEgmMifWLBtItLGns7HVnNI1wXCdecFQS/+3HMXSzD4Pm6psPDvJX67sH1n0TZB28Fnsh4i0t6NV61gTNYyuSUZWQ8hfH/ZMIeqOYvjGehqYMFuSGwiGmq8ZpHQdKybWCgQHhVCYNoFLHc+O1GWFHpa7iMabiRbGsCyi/hL9LFsrKYQ+2KsK2I1dSZP3wsTPPp/3kFpysAuO4TiGvHmMMefHGLy9OKLRNHC6NEUsqwhK2p9og0eRBdeF34mKzqSoiDLFxEIWalN2CFWwqxkRWPTXU8EJMoLiMIFwlCf8FwHzzXxHAvPMfHcS14dC8+x8P1gYrSrxTUVGpQkOZi81TCyoiCrofkxqY2HumB8QshqaJ5MyUr9oSDJKrKq1cbnSjuWad54Fw1d+y4iUi6+5wWvtYeC71gLxsd1LPwLf9fHyg5M3I6JVclxsy0R1eIUc8NHadvx0MpybyQJVQuTbt9Jqm07djWPbRRxjCKOVUF4Ts3QoiArWv38BOdBQ1G14LypIRRVr4trjp167roQq4ieQVciFM3JVZduKc0Okhs/TfOmAysLuJUktHCcxp79NHTfVhubArZRwLWNwM0hUSf2F65ZWdUDYlX7LCBYwX3ruzbuker1IVaSHOhwaeGAKF0gdtpF840aXvhZrZ+SvHC+ufAa9PvKSLfvJNG8qXYfXTTfeC6+7yE8B8+1F88vF88/joXv2YGOnmNhVbKrO8eyjNbaHMRTHTuNcFy07g7Uxgz20OjCbV0PSVWRI2E8cxXWSAH28BggBXFeR0+D6yKFQ0iaNm9FuwookkYq0rnA2qwpERpiGxcRq1sI4Ds2pd4TNN7/Y3T9xM/h2xaypqFEolT6z2KMDS35O9MpU6iMEtUbkKQg7qmn+V56x3+I7V7dMy6sp9nY8r56ZrIQHtnyEP4CK5iEpsWYyB6nb/TZ61Z+al0RK8/2OfTFXiZP5ei+s4lwUmf2fImRwzOMvD27ZEmbVNt2Ntz+cbRQHElRF+pW1Of2hZP8WmRZSJJUJ21LYenK8GLBCzXvvqgJho6eeIapvlevuW8XoIUTbLr7J4llOoPJXZK40tjAtY/PhRgjWVk+SHPx+IhL3oradoHieXHqHOcPfnWNXEbXACGYOv8GqbYdRFKtKx4rSZKQJIVwvJFQbBmhviV+s0wXrhtakztIxzo5MfIUrljdWHuOyVTfqyRbthCOr+wY4cL1ohJONBGKN65o+5sBVY/Ss/+TJFu2XHQ/AQvSudf+fgLq1uPlsPL7SeB7LtXCOH2vfgnHXEXMqudh9w8TvWsfmc8/gTAt5Hh0yXJQzsQ0vmHS8NOfxs0VcGfmKL90EGFdKRNUYI9MUH7lEMlH30/0wG0BSdN17IFhij98GZaypq0AvvADcclwS/2ciNpnt7AMRCBCas1Moje2ICkqwnVwinns2allkwU832K6cIbm5HY0NYIkybSmd+MLj4GpVzDslcutSEjEI61saXuIZLS9/rnpFJkp9C7Y1hce47Pv0N54O63pXVSs7JKuwAtCo1eLdUWsIEgTH3h1iuFDM0iyhPDFZWsEyoqGGoqteOV3o7D0ZHopqam9lZX66nttOyGj6lFUPbK27a4BFo/PpYNy4W1tbPTIFV1vNwpmaZbxMy+w4Y4aoV9lv256+vRlIMsqrmdddSxEJT/GZO/LdN/2sUvIx8rwbh6bC4H36/t+CkiaqkeXjBP0imVKz7+2bN1Os7ef3Fe/i97dgXAcrMFRJE1dFMfjzsyR++qThLZsBFnGnZoJ3ItCUD10BN8w62484XlUDr6DX64Gzz/Po/zCa9gDw2hd7UiahlcsYZ8bqJMqdzZL6fnX8Csrd1e7vsVw9jCaGiGipxHCI18dY7Jw8vK/82wKxji+uPaMQ0XSSEXb1zRL7rpDCOy5aey5pcuDLYdceZi5Uj+t6d1BnVBJpT1zG9FQA+PZo2TLQ7Ugd3eRZUmSFBRJJaQnaUpsoT2zl3iktX7Ner7DVP7UoqLOsiSTSWwkGW0nHevCF15QR/CSdcd0/gy9o8+sfixqWFfEStFktLCCWXYWkClZlQgndcyCvWI9nlu4hesHQXb0GKFoivadH0DRwu9qQrAaVKxZ4uFmNCWM7V2F7IMQzAwcIpJopnnT3aCo75mx+acAv1yh8vKby2/geVi9/Vi9/ZdvSAjswVHswdFFXxlvX+J2832Mw8cX/tx2sPoGsPqWVtn3svnL93PpTjFXOk/ZnEZXY/jCw3JKON7lxacNJ8+ZiWdrit/XhrCW4LbuJ4iHm665rXc7HM9gcOZ1ktF2Inom8PJIKplYD8lIW60w8zRVK4vtlvF8t07AQlqcWKiJRKSNsJ5EltSLrIyCfHmYsbkji8juhe+q5uxl+1Yxl65isVKsK2LVvD3Fgc9v5YU/Ok55ej4DKdYY5uFf3cuhL/UxtUSc1S3cwo2G79pM9r2K73t07PogWmhlmT7vduQrY2RiG9jYfF8tW2qh5cqw84syqC6F55iMnPgBvu/SvPke1GXc5bdwCzcaAoHpFDGdlYuIOq6B7VZwrmahcQlkSca9RnHK9YRidYLe8R+ys/MjhLRkLSRCQlXCJCJtJCJtwGJX9vJhEIJCdYy+ieep2ovJkcBnKre8IvxaYV0Rq3RXjERbBLuykIWaRZtYY4iGjYlbxOoW3jXwHJPJ3pcxSzN07f0w0VQ7kry+JUGaEltIR7sIa0laU7sWrQhPjv4D+epiK8SlcK0KI8f/kWphkvYdDxFJrjwe7RZu4d0ExzPXTGhSIK64MHlvQTBdOIvrmWxqfZB0bAPKEmWxrjQ3CCHwhctMoZdzk89TtS5vcdKUSGDpkrUlQ0gcp3LFNi6HdUWsFE3G9xbHVHmOjxCg6uv7oXUL7z0I3yM3dhKjMEXz5nto6NpLKNaw4jT6q9onXLdsl2xlEMPOLft9xbq8if1i+K7NzMBblOeGaN1yP5nOPeiR5IokO64e10ff6xZWD0nRaLntIUKp5hu+b+F75AeOUxq59sxZ17eWTNm/GgjhX5OWvRx56gAAIABJREFUkxKK0nL7B9CiyTXpz2rguzazp17DzK5WMkGQLQ9SsbK0pHbQlt5DLNyEpoSvqAfoCw/HNSgZU0zmTzJdOHNFYc9IKMOOro+QjncH2dayUiPGMpIk4bomE9lj9I4+u8rjmMe6IlbVrEUorpHqipEdmI/YT3XECCc0jPxipu97Dq5ZDtLW1zkurRQv6QpKSMOr2nWhvFVB+Lh2dXWZP6uAHNKQw/rCRCjPxy2ba/5w85xrLMItSSixEJKy8Eb2DBthryIoVQIlGgpUm6vz58sszzJ28lmyI0dJte0g1badaKoNRQshyVcXZySECEo2+B7Cc3GsMmZpBqM4TWGy98oNXAUMO7+qjJ0rQvgYhSmGj36fuZGjpNt3BlmDiWYUVb/GsfEQXiA74BgljNIURn4qUHO/DhBC4NrGdbufbiRcu7pM5vLaQVIUkt07iXdsua77WQq+62AVZtaGWHnmmi1kAmJ19RYrWQuR3rSPcKZ1TfqzGrhWlcLQyasgVgEsp8jo7GGmC2dIRTtIRNqJhhoIqXE0NYwsBXTFFy6OZ2LZRcrWLMXqOCVjKqj7dwW5HQmZ9oZ9pBMbmMqexHJKdDbdyVTuJK5n05TaRsWcZWTm8FUdwwWsK2I13VvALNq8/1d2c/IfhjHyNuGExu7HunEMj6mziyf8wmQvZ/IT7wk3g2POB0dKqkzLx/bT+NBOxr/6Ovm3+ldNVhyzRP+bX79u1pOmH9tL2yfuQVLl+vgbw3Oc/y9P4Rtra+72XAfPufrYBDUZZuOvfJhIT9MC0/Dol14m98rZFbejNyfp+eVH8comI3/9Ek52/pz5nkMlN0YlP87MwCHCiSbijT1Ekq2Eoim0cBxFi9TIloIkKYAIhPZqukSea+HZJp5j4FgV7GoeszyHWZ7FruRwbQPPWbuJ/kbB92xKMwOU54bRQnHCiWbiTT1Ek63o0VQtszfQU5PqFetFbVwCYnlBn8l1DByzjFXJYZWzmJVZ7Eo+GBt37Un9Bbh2lcG3n0RRV1br7d0M3/dwrLURXH4vwvcdDDsQ2zSd4toRK3zca1QfX88QBPIW04WzTBd6UWU90KCUlXqZGiF8fOHh+faqSagkyaTi3cwV++kdfRZF1mhKbWcqd5pCZYxceYhNbe8nrCUw/qm4AiuzJi/+0Qne/69288F/sw9JkhBAYbTCi398gtLUYqtFIHx39TV/rjckTQmsTf7qJntJU8jcv5XEni4Se7soHBlEOKvz8wvhY1WuLfvhcsj19iK/KqOmooQ70sS2teFHLczSDF713RWg6dsepdNjeIaNmoqQ2NWFlo6ixlYncRHuyJC+azNuyWD6H48uIFZ1CIFjlnBqZV5Amhf+rOl7SZI8T/CEH1hgauTKd238muAj64xAXQnC9+oioMXpcyDJNWHNQOhTltWgMkLt3qdWZPWC8K7vOfiuXbPu3mCXn/Cxq8u7SW/hvYNcdZS3Br4MBNltgrWyWIlrLuvy3oEIAvnXMphfktCUMPnSMJ7vIMsqvu+gqRFAUKpOYDklmlLbyJWv3rK9rogVwNTpPE/+xkFSnTH0iIpjeRTGKosC2tcDtMY43T/3MNmXzpB/8/yqfitsj7kXTiNcn/yh80uXmLjJKJ8apXwqCGROHdjMtt/69E3u0fLwDZup77wFBEVmt/3Wp8ncu3XV7Rgjc2RfPYuTr2COrpS0imUXAEpYJbEhXXenulWX8mhpxZxBCavEO5NYBRNz9uqzliRZItqewDUcrOw1ul1XiHBjFD0VpjJexK4uv09JlYl3JvEdj8pkedWLlFu4hdXC8+0ls86uFUJ4lIxJZornAK5ahfwWloEQQSkdLVb708N2K8QjLcwW+mrub1H//mqx7ogVgGN4zJ5beTrsuxWxLa2k9m+idHxk1b8Vns/Ms8eZe+4kvuPd7Eout1CDM1em/w+/H8RIrwHZjbUn2P/r70eP6+jpCNmTU7zx75/Fs1a2kEj0pLn3Pz7K0NO9nPmbtxd8J8kSSljFrV55hRxpjnHvbz/C3KlpTvzFQTzj+i9kNj62gw0f3c6bv/sc+TPL1+4KpcPc+RsPUZksceS/vrKi47mFW7gYkgQf/3CUthaFv/lqCccJakg//tEombTMX3+5jBDQ2CDz6MMR9u3WURQYHfd45rkqff3B/aDIsG+PzqMPR2hukhmb8PjeP1YZGFrZ/RLUJTzJZCEoXOyvUbbhLQQQ+BQrEzQkN6EqITzfoWRM0ZbZS6k6iaZESEY7Fim2rxa30uhuFmSJ6JZWlPg1qKl7Pr69Xos5v3chHG/NLIilkQJv/PtnOfz7L1EZL6Lo6oqKgF+AJEuoYRVZW5xpl9iYZsdP34ESuvL6SgiBZ3l4pnvDSLykySjhlQWvKyEFRb+e2YS38F7HgTt0Hv1ABFUJrjdFhnsPhHjkoQiSBKoKv/CFBD/7+TiOI6gagr27NLZu1upe+wN3hvjPv5VhQ5dKqSw4sD/E7/12Axu6Vn5tCnx84dakTG7N7WsJIXxmC33ky8NIkowQHrP5PpBg76Yn2LHhowDMFvquaT/r0mJ1MdRUlMTuTsyJPMbQLEpUJ7KhCS0dxbddzPEc1lThsu4BJRYi1JZGa4gh6yrC8bCzZazx3ILMroshh1QSe7vxLZfy2XGE46E3J4h0NyGHNXzLwZ4tYY7nFsQ+qckIenOScEea1J2bkFWF2Pb2hfsRAmNkDmNoYeq6pMok9nSjJheWzKicn8IaX2FshyyhNyYItaeD+CFJwjNsnGwZazKPf5ElRImHSOztRrg+pRMj+KazqK3oxmbCnQ0YQzMYw0uXubgqXOhnawo1EQYJvIqNNVXAmikuW7U+1JEh2tNEdXAGayKPpClENzajNcYBCa9iYo5mcXJrZ2LXmxLEtrcvyCj0DHvpMVslhOtjTJfxHW9tLTESZHa10HJ3F2e/fASuEMZgzFY59J+ewzWcgFy9G3HrGbQySBKJvV2ARPnM2KpjM/+pQlMltm/VGBxx+eP/USRf8InHJJza2laW4Zd+JsGJUza/998KFIo+Pd0qX/2fLXzw/RG+9LUy/nsrJHJdolgdp1SdrGvwlYwpTg99n9bMroB4Fc9RKF9Zi+9yWPfEKrqxic3/5jFyr/cx+9xJ2n78buI7O1BjYXzHw54pMvvDE0w99TZeZeHTQ9IUWj52B40f3E2oJXiAS6qCcD3cskn1/BTj3zhI6djwoklbTUbo+aVHcMsW53//uyT2ddP+E/cQas8gawq+6+FkK5z7vSepnJ0AQInqbPrfPkp8d2d9X5Ik0frx/bR+fH+9beELxr7yGmNfennBPuWQRvtP3EN8dyeyriJpwe8H/+wZpibzV4wtCXc20PLx/aQPbEJrTKCEg+wl33ZxSya5g+cY+vMf1klLqC3N5l/7GG7Z4ux/+PtF5E3WFJo/tI+2T9/N6JdeZuwrr61JfEu4u5GOz95HfEcHWkMMJRLUzfItBztbIfd6LxPfOIhbWBx3kz6wme6ffYixL7/G3Iun6frCg6QObK6TUd90yB08x8B//f7VSVQsgeiWFnp++VHURDg4L7KMMZ7l7G8tHrMFx9kQIdISR1Zl7LJNZbyIb1/DQ06CUCZCtDWOJEsY0wF5vNigKSkSkaYYkZYYHe/rQY+HaNrXhlsjS3bBpDgw32c9FSaxIY1UW8Wbs1Xssr3gPOupMJGmKJWJErHOJMITlEcLyKpMoieNW3UojxUR7vx4S7JEpCVOuCECsoQ1V6U6VUZcev2I4LgizUGfhQ/VyRJW3lh0TwohkHWFZGscLR7CKVtUJkrLEkE9HSbaEkcJqQjPxy5ZVKfK13YO1gEkVabj8w+CBOf/nydxC9euGH5V8H3MwgxKOBYUkVY1JLWWvPEuLHVkmIKvfbvCv/7FJF/682aeec7g+89W6a+5+RrSMnt26agK7N2l18lWS4tCd6eKpsIVa0xfI4TnYuamAJBqYyorwbgG2bTvrjG9HGRJRVV0JElBqpnpXc+6ZmV6IfxLkg0EhcoIhcrqQ3KWw7onVheQ3L+R2NY2nEKVyW8dwqtaxLa0kb53Kx0/9T7ksMbY376y6GEa39WJloxQPjVKdWAap2CgNcTI3LuV1J0BAen7nW9hTV4i5VCb1PXGOM0fu52mH9uDMZKtBZL7hNrSqInwAsuIb7lMPvkWyg+Po0R02p44QGxrG1NPvU3hncEFbZtjiwMjPcNm6C9+hJqKoMbDtH36blL7N65ofKKbWtj0qx8ltq0NJ1eh+PYAxmgWEIRaUkQ2NmONZZe1BN1IqIkwiT1duEWD4tEhzLEswhfEtrWRvnszbU/cjXA8Rr/08pJWCklViG5tJb67k2hPE/lD/dizRZSwTnRzC8bgzJqRKoDSiVH6fuebyBGd6MZmun7moctuL2synR/YwrbP7iOUDsq5CF8we2ySU//fW1QnrkIHSZJof2ADO/7ZfmJtCTzLxZyrMn14bIHae7gxyt5fuofEpgzxzhSyJnPH//5gXbNo+q0x3vkv8+Oa2tLAnl+8m3AmSqQtztA/nOXon7y2gKw07+9gxxf2M3N4jI4HN6KEVc5/6wThpigdD2wE4NifvcHYi/0gQI1qbP/c7XQ8vAktqgXWSNNl6Ole+p88hVOef/pIskTXBzbRsKeNSFMUJaRSGS9y5m/fYeqNkQVETE+G2fMLd9N0WxtqTEeqHc+pvzpMdXJ+TGVdofPhTWz9zD6iLfG6a9WzXY792RuMvzRwy/p1A+C7NmOvfgdZCyFrepD1qelBBqimo4SiqOHowtdQFCUUQVI0ZEUNCFiNQCjatReplyTq14OsSIT0hUTkmecMjhy3ufeuEJ/+RJSffCLGH/xxgX94pookBb//3g+qvPCKuWBBMzLm4tyA0D/XKDH0/FdQLhlTWdWR9XBt/KKooQhKOFYfV0UPzxNaRUWWleC83EDpEAmJkJYkFeuiIb6RaKgBTQkjSwpIMhIwMvsWI3NvXVc5GVUJI8sq9jXUfnzPECu9McHs0WFGv/gSzlw5WL2GVJr7p+j6wvtp/tA+5l48jTE4HwQrHI+xL7+CJMlY04UgNkYIJFki/+Z5Nv/ax4h0NxDb0b6YWEkX9hun8f07mfjWIWZ/dALfcACBpMjIYR23NG9VEZ5fD1RXkxEaH96FEIJq/zT5N85d+SB9ERCuMZB1lYb371zR2Ei6Svtn7iW2tZVy7wSjX3yZ8umxehyQpMioiQie+e4opVA9P8W5P/ge9nQBt2TWSZAc0mj75J10/NQDpO7azMS3D+GVlpDSkCB992YqvZOc+73vUh2aBd8HSULW1/6S9yoWlXPBKtEz7MvHV0nQcqCLvb98DzPvjDP8P/pwqzbJTQ1s/9zt7P0X93Dkv72KXVidREi8M8meX7wHz3Q49qevURkvkdyUYfOn96DF5idHK2tw4i8PocV09v7yvURbYhz63efqbkb3khiq2aMTvPobT5PoSXPPf3hk2f1HmoIMvnf+8BV2/cydbP3MPkZf6OfwH7zEvn95Lx0P9jD15giu4bD5k7vZ+ImdDD51munDYwgf2h/sYetn9+GaDv1Pnq5bt0KZMI172hh46jTFoTyx9gQ7Pn8Hu3/uAJWxIqWh+fuyYWczxkyFo3/0Gp7t0npPN5se34VjOBz709cRro8kS3Q8uJHb/tV9FM5nOf7NExhzVdRa9mSxP3uLVN1A+K6N79qwZNJnwFSkC4xFkgLLhSzXiEMgUaLoIaKtPXTc89hV90MIqBqCVEomlZAxDI+WJpmd2zSyudr8I0M0IpHNeTz1gyovv27yx7/fyIc/GOH5lw1yBZ++8w6ZlMzxUzZzWR9FhmhUwrTEDUtW9R0LfzlNvxpzvDCmwauMJMnBeOqhgGRpYdKbb6Np9/03pM8hLUFbeg9tmT3EQs21sjbSIgubpkZZLshUV2N0Nx1AUwLvRKE6xmTu5KqlMNob9hEJN9A78oOrORTgPUSsvKpF9uUz2NPz2YK+4ZB9tZeGB3aQ2NdNYk/XAmIFYI4sUajRFxhDs1TOTRHZ0ESoZYnyABduElmieHKUmR8cWyB6KVx/QbzSzUS4I0Pyjh7cqs3ktw9RPDK0wD8kXB/beveoRfuWS+XM+OLPDZvC0WFaH78LNRFGjYWXJlYExzT5nbeo9C1UAfZucjyJGtXZ8JFteIbD6b86TGUsuF5zZ2ZQdIXdv3A3jXtbmXh1dRoqzXd1EG2JcfRPXmfkuX7wBbkzM0SaY4FcQw2+61OdKKFGNdyqg+/4lMeLuJWll9PCEzhlG3Ouiu8uP0FJssTkG8NMHxql+Y52EhvSjL3Qz+yRCQof2hq4PHWFcERjw0e3MfP2GH1fP163TpVHCzTf0UHXB7cw+nx/XdJBeILRF/sZ/kEfwhfkTk+jxXT2/OLdtBzoXECsrILJub8/TuFcEOtXGi6Q2JCm7b4N9H3tGNWJEloiRM/HtuNUA7J1sdsTWVqzRBC9JUly/ybyB/tw81UkVSHzwA4kTSH36ll8w0aJhUjfv53KmbFAmqMWVxjb0YGWjuJVbar9U5ijc3WiKakKydt7QILy6TFiuzoJtaQQno85PEvl3OTCmCkJ9KYk8V2dKDULeuXseLCAvNQtJEuEWlJEt7bV91/pHceayC+w8CqxEOl7tlLtn8KaKRLf2UmoLT3fh96JNbIIB+WHLj4lF976tnnJltd+3t45ZvGZT8X49X+d4kyvw/atGg0ZhbkasUqnZP7FzyTwfZiZ80gmZDZ0qnzz7Qq2LXBd+Msvlvjt38zwf/9mhrPnbMIhmZZmhb/+colTZ98F2ao1OYELY3rxqHm2gXNR6GkofWNKDcXCzWxt+yCNiU0osn7V7krPd8jEemhIbEQIQcroIlcexnQKq2hFIqQlCKn/BOUWloJbNJZ2nxUNqgPTJG/bQHRLy9I/liS0dBQtE0OJhZA1FTmsoqaiAEtbOWrn3jedIEh5jZXE1xKRrga0TIzKuUlKJ0fXTRahpKvojXG0VBQ5rCGpCuH2dKDk7slI6vJJreZ4jsr5qRvY25VBj+uktzZSHMxRGZ9fBAhPMHci6G9qWyOTB0cWxCRdCentzThlm/zZ2Xr8k/AF2dMzNyTYXPhgzgSzsl22cSp2nRy5hoNciweMdSaJtiXInpqm+c7O+WtRAoQg1pFET4Tqv3UqNsX+7LzLT0D25BS+65Pa0rigD+WxAsbM/JPBLprkzs7Q8f6NhBsiVCdKhBsiJHrSzB6dpDh4SfzbGpoUtIY4HT/1APZUgeKRQbSmBB2ffwAlomMMTlM9N0WoI0PH5x9k6E+exhzPkbyth47PP4DWlMCrWsghDeF4TD/1NrPPHEW4PrKu0vjIXvTmJMbgNPE93eAL1GQEIQTjX3mV2R8cq49ruLuRnl/5COHOBtyigfB9zNE51Hh4kUhv6q7NtH/ufWipKJ5ho0R0PMNm4muvk3+9t06W1FSUts/e9/+z9+ZhcmXlmefv3DX2yIjcN+27VFWqfacooMxiTNEGbLAxBnvanh7b03TTj+2ZP9zT0555PIzHbjzThsFtmwHbDTa0cWGqoDZqX6VSVWlXSinlvkZk7BF3PfPHjQxlKFPKTGVKpSrq5alHScRdzr0Rce57vu/93o+5506iRkySt2wJWkKFDYpHRzj3n9ZPv3g18exLFn/4xzk+8kCYW28yePr5Ggdft+hoV4OIVkUyeM7l3rtC7N2lU61K/uKbBR5+rIpd50zPvljjd/8gy89+MMwtN5pUq5JXD1lMTK18QacKHVUNosyOZ61bg+drESE9ya7eD5GObVy2L+By8Hyb2eJp0vFNwVxjtpKIdOMUa+haGKvukK+r4YueSxFqPSq2NrxjiJVvu0tW8Pmuh5MLxJlG6gIWKgTRnd10fmQ/0e3daMkwiqY29Cbz4u4lQ4/1Odh3ApH6NQsh0NviCEVgTxfWXKV2NaBGTVrv20P63p2YXS2oUQMIoglCCXr62TOXjrC5+Qq+de1dqxrWUQwNu2AtSjm5NRev5mDETYQqkKvgQ0bMwHd9nAt+A07RWiwIvxKQEn/+YSol0vPP95qrC9ABtLCOamp0372R9ht7Fh2mMlVqWrFK18etNn+Obs3Fd3y0uj5rHl7VbY6qSXDLDkJVApsKQI8aKJpCLVu5oik/e6aAX7MxupIgBOG+NNLx8FWXUG86IFYdSaQXFNiEN7TR9+v341VsBr/8EPZ0AS0Wov0j++n97D3Ys0Xyr9TlAgKiu3pwcmXO/vEPcPIVjLYE/b/xfroevJXCa2expwsopk7PL92D2d3C0NcepXxyAsXQaP/QflJ37aR0/HzlU6i/lf5/+X6qQzMMf+0xnEwJLRGm93Pvofdz92LPFpqiyIqu0fbAdeQPDDL4fz6EV7JQ5otM3gZzzFKo1SQPPVLhoUeWFvPXLMnff7/M33//4vO958GrhyxePXR5AmuBQl96P73p/QCcGH+UbHn1DuACQTu9WFTJE2hpVTR62UKr6KRKhWF5igpvXaZCU0Ns676fdGxT4zcvpcT1alTtPJZTwJce6fhmdDW0omPOlYZxPRtNNVAVnXRsE+nEJtKJrZwee5LZ/AC7N36UWPhi0TiBoUfJ5FcgzbnUta1p72sJC0SHl97o/J+J/RvZ8m8/ghYLUTg8zPSP3sCarFssCEHXgzeTumP7pQ8lJdd0Da0I0jTAFWiqKgKzl3WEYmr0/9p7aX/gOpxsibmXT1M6MY6TqyBtF7MrSf+v37/scaQnr0kHbq/q4FkuRiJUj9Kcf08LaaghPSBD3ir7PpZthKaghZvFplpED1JcF8F6pFAWHGxZeDUH33Y5/Q+HGXn8zKLvpPRkUPFXh9AUtIjRtI0W1lF0JUhfygteXxjFFAItZiA9v2Go6lYdpCcxW8KL7v96wpkrY80UGvYroQ1tWNN5fMcjvKkD5ZUzjSiSnSnS+fFbMdIxzn7z4QaBcbIlJr/3CrF9G2j/4A3kD5zvzuBXbSa/9zKVwenG+eaePUH3p+/CaE9gTxcI9aWJ7+kj+9wJ8i+dbkSRZh4+ROrunU3jbbljO2rEYOr7r1Kpp8+dbInpHxxky+89SMut25qIlVAETq7C+LdfwJm9dmQEb3cIoRA2UsRDQXZFVYxl9lgaBiE2iV1kmaYgs0igm41sEruoUSFCHFOEOCJfxuOtkKwIOpI76EjuapAqx60ylT/BRPZNirUpXK+GocW4ddvnVkysak6eqp0jHu4ABPFwJ9OFkxQq41hOESEEYTOF5ZSoWAHhbB6VQjLat+are8cQK0XXUCMmTqZZyS80tVFq7ywoK1YMnY6P7MdoizP98OuM/e3zOLly42GshPRF9gxNuPae2UtDSpx80KneSEURSxhFXnp/LkpYhSoCj6l1RGRLJ63v2YVbqjH0F0+SPzBYN0EN3vdd722ZZpiHXbTInZwhtaudaE+iobESqiC9txMhIHcmu6o0IEBuYJbe926hZUcbhbN1AbYQtOxoQzUXf+ZSSnzXQ9VVVEO7qMZqvVEaK1CeLJHc1srZH5xoEukLRSAU0XTtetQguTXNzGtjAdkUkN7TgaIp5AebfdOifQkiHdHGMY2ESWpnG5XpErW5gKzVMhVKo3nSezqI9SUpjSzQXwiCHoTrQch9SfXcLOENrWixEJFN7VSHZ/HKFrHdvWixEGZfGmssi3Q8Qr1pvKpNbbxZzuAWq1hjGUJ9bWjRUOO772RL2AsJjZS4hQpCVRBa8Hnr6Rhq1KR6dga5YPHn5Cs4c83zZGRrJ4qh0faB62i5Y0fjN68nI6ghHb0tFhD0+TQzQeXykr0w38VlQwgFVVl7JZ6OgYZBSeaQSHQMusRGskxzRh4mQZqt4joixCiSW/6A6wxdDdOZ3NO4VsercW76BUYyB3G9hfq5VS4w3Sq2U0SG2hFCYGgxZvKnGJ09iOe7dVE8jM4cDNzVL1jYCUVlW897Md7VWAXQ4iHMriS1kebJVo2YhDe0IaWkOnReuK6EdMK9aaTrkT94dtEEoUZMzM7kxU+4VjsQKRsT+PxEeEUgwRqbwytbmN0tRDa1B+L1lezq+kjPR9FU1MjilZMaCxHuS6/rcM2OBGosRPXEOMXDI4sKAMyulkD7cSnSew3DrToMPzpA6w3d7P78zQz/6BROxSaxKc22T+xl+uAY2cPnBfeBo7hGqDWCaqhIAssEu1DDtz08K9BfTB8cZ+snKmx5cA+e5VIeLxDf0ELP3Zua7BbmIV2f0kie7js30P+BbWQOTyIUgVt1zgu6BaimhmqohFojKJqCFtEb5/csb9WeT7VMhaGHT7LjMzew59duZurlUeyShR4xiG9soTJZZPy5c42InVAEPe/Z3PDXinTF2fLgHipTJaYPjC24IDDiJts+dR1DPzqFV3PpvL2ftuu7GXtqsKG9sosWQz86xfW/dSfX/9YdDP94gNpcFdXUiHbHyZ/ONLRua0X17BTJmzejt8YI9abJvXoGr2LRet8e9LY4oe4Ucy+cDKpVdS2oJr0wUiklfs0NdIXzDdsJJAgXGnteGJAWmopQFTyrObInPX8RcVdDBkJVMbtaMC54r3h4hNpwpl5Rdj69Ky33mowKv50hEOtCrES9qYpdd/6N0UKICEPyJFXK9fclBmu3qLgcRMwU8XBnsJCRkkzhDKPZQxeQqtXDly6We/5ZrmthFKE3GltLKanZeWp2Hn+pZteei+taa9ZZvWOIlRoLkb5nJ+VTE+eNIxVB8saNRLd14hZrFI8scFP1fXzXQygKat2BfH5mEppC6s7tRLZ2XvyEa5xPfMfDK1tBc9uNbU2rwfVGdXiW8skJEvs30vXgLdiZ0iICKtTgPriF82kYJ1fBLdUwu1qI7+kLVr71iV3oKum7dxLqXV9iFfQ9DKwyFFNnYR220ZGg9b7dqBHzbUuskDD92jhHvvoyOz7kKRasAAAgAElEQVRzAzf//n3By74ke3yaY//lAFYumFyEpnDjl+4Nok6GRrgjWEXd/eUP49semSNTHPn6KzhFi9JonmN/8So7fnk/+//13bg1h1qmwtgzZ9Giiydq3/EZeeIMrfs62fkr+wOSZnuMP3OOI197GQAzGWL/v7mH+MYW1JBGuDNGVzJEy/Y2fMdj8uURjn79lVVf/+BDxxCKoP+BbfTcuzmIUvkSp2xz+h8ON22eO51h/JmzbP7YbsxUBC2kUZ4ocvJbrzVFm3w3uB7p+ez/4t1oEQMhYOqVEU59+80GkZCeZOzpsyi6ytaf38sN//ruxv33LJfDX3t53VKE1dEsQlGI7epFaCq1kQzS9ZCeR2xXYBJcOTsd6MgKVZSwgRJq/qyEqqClong1G69qnRfdrmB8fs3Gtz30ZKRpflMMDcXUmjSITq6Mkysz8pc/wZpaHMGQjndNeNy907FeEavAYkCiEphrtoueut4qU3+/Uda+5nNdDmKhdrR6es+XHlO5Yzju+hjV2m4FiY9ARVX0RpQqOJfLyZFHsJ2Ln6tQncT21jaWZYmVEKIf+CbQSfBz/rqU8itCiDTwHWATcA74BSnlnAgSpl8BPgJUgM9LKV9b6tjrCa9skbxhI8aXPkru1TO4hSrRHd3BgzhkMPXDQ1TOno9YeVWbwqFzRLd10v2J21BMjerwLFo0RMttW0ncsBEnW0LpTi19wjVGrHzLoXRslPQ9O2m9fy+e7QbaBgFaLExpYJLyyWbLAT0VRY2FEJqCGjYaVYtGZ5Loti582w1c4wvVJoLkVWzG/usLGB0JWm7bSqgvTeH1c1THgsiE0RojsrEdayrPuf/8aGM/N18m98oZuj95Gz2/eAdmd0tDAJvYv5HYnl6s2SKR/sWrHsXUMNriwapZUwn3pRFCoIR0Its68Uo1pOvh1RzsTKkxaVfOzVAbnyPc10r/F+4j88xx/IpNeGMbre/djRoLN3mDrRf01hhqxAzubUhHiwY/erOrhci2ziB6Vy+E8EoL0le6itEWR9E1hK4Q2dgWeJgZGpHN7aghPbDesIPrDPoI+ow+eYbZ18cJd8VRNAWnZFMeyzciUBBEFk5/7wj6EtFCALtknRd2+5KxZ86SOTLVcF6vTJWwcjWmD47hlBcXdpSGc7z8vzxBrC+Bamh4ttdUVeeUbU59+w200NITvZWrBlHXszHGvjJHcTh4IM88O01schNuNvhMT3/3CFpYxy4GZNirupz69hsMPzZAuD2KagbnrmUq1GbLjWjV0MOnGPvJIMWRPKNPniHSEUP6kvJEs/O6na9x6E+ew5qrYhcsYv1J9JgR3NPxwqKqSK/mcu6fTzDxwlDDed13PKx8jcpUad3S/G62hFusktgfzCXOXLnxHUrs34hXsQN7GCkpvDlEyx3babltW0DAPB8ERLZ3E9nUHvwOqg5qZOURhtpYFjtTJHnTZjI/Odr43oY3tGJ2tTRF8PMHzpC6awexPX1Uh2aaomFK2Fi33pfv4tIIiNXl6aoWwsbCoka32ExExumgj2F5CqcewdIxECj4b4m+Ckw9gVInPI5bpmzNLrPHyuH5TkPCogh1UQVgzS4svWMdM7kTax7DSiJWLvAlKeVrQog4cFAI8RjweeAJKeUfCSF+H/h94PeADwPb6//dDny1/u8VhTWVZ/IfX6Xjw/vp+9X3BNEO3w+c2H9wkInvvtw0OUjXZ/IHr2F2JUncuImNv/l+QODbLs5skYnvvYJvu2z47y4ilF7r5Csh+/wpzJ40bffvoetjNzeiVl7VZuQbTy8iVr2fvYeW27YGK876qhOg+1/cSudH9uPbLr7tMv3w64x/56Wm3EDx6CiDf/owXQ/eQmxPL20PXN+wK5Cuh1exqY5e0OtPwuQ/voqeipK8aXPQdufnbgp6Kc4Wmfqng/iWw8Z/9cCiy4vu6GbLl362MU7F0BGaQqg3xc7/+ClkfazVkSxn/vifGwJYa2KO0W89R+9n7iJ153bS79kFnh/03js2xui3nqPjQzcQ2927xg9gARTBhl+/n/h1/U3jBej+1O10PXhzcG8tl4nvvcLUP7/WiC6GetNs+58eRIuF6vsFrYbUWIhtv/cxfCu4TnumwOBXHqG6gNzXslVq2UuQRElgn7BSyCDVVss0r7bmfZ2WglO0mDs+g1B1Upuuo33bBiqpMTIDr+I7PnPHZy667zxEWaM6YDdMRosTGc7MHcCuBtdWGl3CR0YG7XFqsxdfGS50S69Olxstei6E7/hN11gYXGy7shSsbLVh63Al4BSqOPkK8b39ZJ89HiwIpMSazJF+z+56dD24/sKhc8y9eIr2n70RLRmhMjiF0ZYgdc9O7EyJ2UdeX/X57dkimccP0/2Ld7Lhv/8AxcMjqGGDxP6NQQRvwRxWeO0ccy8O0PUvbiW6vZPKmWmklJidSfR0jInvvND03X0XVwbrlQq0qTEmB9kkdpEUafJkmGS4EamKkcDDpba0M+sVh6oYjXSl5zu43vrZFQWmp8Hfb1WielliJaWcACbqfxeFEMeBXuBB4L31zf4/4CkCYvUg8E0ZlPu8JIRoEUJ0149zxSAUQf61cxSPjgY95lpj+JZD9ewM5cHpJX2mnNki5/7zY8T29BLqbgFFwS1UKQ9MUhvNoqcijP3t81QGF2suvIrF5PcPoBgatZU2QL4AbqHK2N89T+7VM0Q2taGYekDs5sqUjo0t2r7w+hD2zKXZNkDp5MRiwYWUlI6PcXY4Q3hjG+H+QFQLErdkYU3MNSqMFsKZKzP0/z5BdFsn4b5WlJCOW6pRHpigOpLFaIsz+o2nKZ0YbzqnPVtk5kdvLH8PSrXmz0ZC9rmTVIdmie7oQk9EgmjCxBzlU5O4pSBSUTw6umSPs9LxMUa/+RzWxFwgel8JpCR3YJDq8PIkpjI41fRrdQtVZh87fElPLQiq4ZbqbXitQPouxclB9GgLkbY+MgOvXnJ7gYKmhQILDJTGPQlC7yZVO4ff5L8j0FQDRWhIJG7dn0cgUFUzeF16uF4NiURVjPq/wUPG9WpI6Tf+vyI0hFBwvVrjPEIoaEoIIQSuZzWarM73HAOB59sNvYWqGEjpB+07hNb03rrcU9ulem6G8MY2KmemGmS8PDBJ8uYt1EYyDYsYr1Rj4jsvYM8USN6ylcRNm/FrDuWT48w++mZD1C6lxCvWcPOVZpG9DKLgTrZ0PtrkS2Z+/AZCV2m5bRuRLZ042RLZp48Fv932eOMYXsVi7JtPU7t/L4kbN9O2vTu4j/kKpZPjuIUF2hfPx82Vcctr08MsB6Epgd9WzblqFg6aUpeFvEXQtXAwhnXANCOUZB4DgxIFbOY/L0GZIoPyKDXeKqughWllsa73XFcjjd6Cvu/g+6uLysXCHehqmLnS6m0u5rEqjZUQYhNwI/Ay0LmALE0SpAohIF0LuxmO1l+7osRq3ufImshhTay8ysEtVC/aTsaeKTL10MEl3/MqNtMPr34VeSH8qk3xzeGg0fMyyD67xhClDCbw0tFRSkdX3r3bK9UovD60pOjdGp9j4nuLdTbWRI7xb794eeP0fKrnZha55M+j8MYQhTeW/tKXByYXua0vCwmZJ4+udpRAUJ018d2XL2vfleN8C4rzjtkS6fsIRa1XfM2LvVWk753/WwbbBMUS9QeuUBqCdukHmjakxCnnsEtzGNFLFG0QkKrO1B7ak9ux3QqGFsV2gwk6HumiO3Udmhri1NijWPV+W4lIN93p61AUDQWF0dnXyFfGiIU76Gndj6roCKEykXmDufII/e23oCo6uhpGU0NkioNMZo8E7SaMFEIEDsn58hgjswcRCDpTu0nFNiJQKFszjM4ewvMdutPXkYz2oAiNmlNgePoVbLdMT/p6FEVH10IYWoxscZDJucv7HlwMU/90gMwTh5sqkueeP0np6EhAqhYsRpxMial/fIXMk0dQDL3eDL5ab5MVwK85jP/X5xGaincBsSkcOsfA2emmQhyvVGPyey8z+/jhoDm85eLkymixEELXmo4RnP9VZh8/gho+36DdK9Waikjs2SJnvvzQ+vjEXUJbGu5vZcvvPMD0jw4z/diRK25srCkhdnS9F1NfotPGVYKqaITW6fw+PqUlK/5kQ2v1VsHxasHCSmioqo6umqwHTVcUnYiZYj5k5Xi1VTdtbktuJ2K2Xh1iJYSIAd8DviilLDSZ+EkphRCr+tYLIX4D+I3V7PMu3sVPIyJtfbRs3IdqhHAqRRRVAyGYPfEi6W23UJwYoDw9hB5J0LHvPqaPPIMQCm277sAqzBBt34jnWEwdfgopfdp33YUZTyMUlfzYSebOHGI1QfOImaInfT1np57Hcops7ronaJQK5EtjeJ7D1u73NkL9AG2JLSAl5yafRwgFuy5UrdkFRmYO4PkWHcldtLfspFidwtRjKELn7NRzRMw0mzruJF8eRdeihM0Wzkw8gyIUdvR+gLnSMJoaojW+laHpl3C9Gtt73k9rIs/U3HGypSEyxUFURWdz590kIz3MFAYw9CjRUBuDk8/iuNVVr2xXAq9YxbtAE+hXbayLdGqQrr/IMqZ5A9nU2H0lx5WOt8hraqEGs2nb+WjUJdam0vOxp1bTJmRpaMkwbfftYvbpE0tGcxVDJdSVRI1fnco1RVFJxzYSNduuyvl+mlG1c3i+g6JoGGqUqNlGsbr2atyIkSZat1qYrwC0nZVH5QQKmhpqErxfDla0txBCJyBVfyul/G/1l6fmU3xCiG5gPo80BvQv2L2v/loTpJRfB75eP/67Nbvv4l1cBKoRIpRsJ3v6IJ3Xv4+J1x8nvfVGwukejHgKNRMI7YWiEUq0IVQVoagkencwU5hl6sjTQNALTCgqhbGTeHYVI9pC5777KI2fxqkun2KeR8hswZMO+fIovvQoVCaImEF1qMTH820uJGoz+dP0td/M5q57yBQHmc0PAEFT1fbkDnQtTMRIIZENsWmuPELNzuN6Fp50MPUYAIXKBBUrE0ycTpGw2ULESBELtdHffisgMfQYphZFVXSiZivJaC+qohM2U2hauD4qQa40Srk6u+pGre9i7YhsaqfzZ/eTO3D2mk6Tv12hoePj4S/4bhuEMAnj4VClvL4GwatAsTqF5ZYDOwRFpSu1j2zpbGPBdTkQQqW3dT+mFswTEp9ceYRYpKPRmHklx4iF2/HWqPlaSVWgAP4SOC6l/JMFbz0E/CrwR/V//2nB678thPg2gWg9f6X1Ve/iXbzTYZfmqGTHscs5anMTeNZOFO3S1UNurUxh7BRO5Xx0QdF0jFgKM7YF1QyjhqIohslqNKy+7yFQEEID6QVpvGXKtovVSU6NPk483EFv201oisl07gSbOu+kXMswNvsaqdgm0vGNzIfxA11UUNmjCLWhpQqEr0FqVFV0fN/F9W3KVobRmQMNbZXtlklEuulvv4Wzk89iu9VF/jS+dN+yh8tVhyIw2+N4VacRsdLiIbRkBCdbqnecCExBha4GBqR1o9mg6bkJAtyShVusLUrNqbEQWsTAyhRRVDVoEaZr+I6Lm68GmkcReASqUZPUrZtRwwZmV6BvhUCXZk1fQPJlUGWsxUMohhboUHOVRT5e6w1Zb1h89SEuuxHxPFQ0top9VGSJUc4g8YmSYIfYT4IUDjZn5TEmuPx011pgOQUyxUGiZitCCNKxzfS13szw7KuX5WWlKgZdqX30pG5o3DvbKTOTP8nWvveRjPZeoPlcGgKBouhMzx1b9RgWYiURq7uBXwEOCyHmRUX/MwGh+nshxK8DQ8Av1N97mMBq4TSB3cIX1jTCZeA7Pk6+ilusXp2eaO/iXbwFkHUdFHJBDz6C10Q9DaeoGop2vqJI+j6+16yDSW3eT7S9n+yZQwBEOzateiwVaxZfuvS27adm50lGe7HsIH0VMlqI1j1qYuGgLUfNKZKOb0ZTTTzPwvPtugg90IrNt5lIRLubSqNb41uw7CJhM43nO1StPMShJdpHR8tuVEVDVQzKtQxVO09LtJ9EtJeqNRecq+IE9wiJqpokjRQhPb7q632nQA0bbPqN+6mOZBn+xrMIVaHzIzfQ/eBNDH/reaYfeRPF1On9xdvR4mFO/+kjKLpG+s5ttL5nF6GuJAiwJvPMPnWc7EtnmkTlbffupPW+nQz91TO03r2DxPX9aPFQvcr5WXIHz6FGTHo+eSuJvb2BHUnEZMvvPNCo2K6cnWHgyz9sdt9Phun/5TuJ7+0LjldzmHthgMl/fj0geFcAUvpM5U9QqF3dRu66GqI3df2anb8NQqTooEa17ukk6BVbCBNjjEEixOkX28nIqQWi9qsHz3eYyh2lI7GDsNmCquhsaL8dXQ0zlj1ExZprLJAuBUXoRMwWulPX0ZPej6YGaWPf95gpnKJYm0IIhencCbLFc8vq9IRQ6EztXfP1raQq8Dku7tr0/iW2l8BvrXFcK0blzCSn/v0/BNqAK+BvdDEIlPpK910y99MGVTNJd+5BXSZiBFApTlHInrsi45C+h1MpEO/eilPJE+/dgWpcOuSth+N4dg3XqhBOdzeiXkIoqKEIWiiKqpvokQSuVUF6iyc3yylxbuoF2pM7UQyF6bkT+NJDCIXW+GYiZpqKNUsqtgFdizCRfROBIBHuAiEoVWeYzh3D8aqMzLxGe3I7yUgvmcJpNDXUmFCL1Umi4Q4UoXBu6kXsuqNyqTaNqcfRtTBDUy9StQNB0ND0S7QmtpCKb8SyC4005eTcUVLRDVTsLGOZ1xvbF6uTuG6Nn5bfsHSDRWh0eycoAjViEN3WCUKQ2NvL9CNvooZ0IpvbKZ+ZBl/S9t7d9P3SnZTPTDH5g0NIX5K6dQsbPn8vSshg+tHDTW3Awv2t9H36DpCS7AsDSM8n1NOCkwtSPL7tMvfyGQqHR+l+8Eai27oY/bsXAy87Aj/CC1tWtd67k+pIhtmnjuPbHqk7ttLzqduxs+VA1H4FFtS+dJnIH2O6cHLdj30phPQErbEt69BSRUNBpUIQ/QsRoZUuphjmrDxOnBR7xW1EiL0lxAqClP5o5iBbut6DIjQMLUJ/2y20JraSKQySr44hfa/hdwVgaBES4S4UoREyEiQjfaTjm4gYqcZ2UkqK1UlGMwcbVb6ZwhkmMm8uOyaBQsRMEzLWVkDwtnde9y2X2tjl2R1cLiJ6C7va38dk8STjxWP8tEzM7yKAqpm09VyHYQa5fIRCONqK7znYtSIgMUJJpPQYGfjJmomVa1WwCrP4nkt1bhLpudTyM9jlPNXsJG07b6N15+3U5qbIjxxDug5S8ahmx89XAtaROfMa7TvvoG3HbVSyE+SHj+C7DnqshbYdt2FEUyiaTud19zN39g3K00uPvVCZoFBZnOEfyxxacvtM8QyZ4plFrxerExSrzceZj1qVarNMLRGSr9p5RqZfWZTCK1YnKVYXV4ROZA8veg1geh2MAN9O8B2XytkZ4rt70BNh1IiB2ZEg+8IA0a0dKGEDLRHGaI0x/ePDGK0xuh+8kcq5Gc7++RPYM4EAfu6VM2z57QfofvAm8ofOYU2dT93pyTDS9TjzZ4+dN9Jd0ApHOh6lE8Hnnb5rG+H+VopHR6mNX1wt71Udzn71SazJIKVdODzC7j/8JMn9G8g8e7JhWbGe8HyvrhW8upDSX6fzBrGQ+QbLSdpQUJmR4/h42FQJnNnX7pl1ufCly8jsAUJGkt70foTQUBSNWKidWKgdX3p4voO2wDC1q2Uf7YkdKIqGppqNopmFqFgZTk/+hGI1kH3PFc9Rqa2sCnJeI7rWJ/rbnlitBm2RzThelby1ynL8C6AInbjZSa46jkD89Gg03gUAdq3AyYN/1/BeSXfupqV9GxNnX6JWCTqmh2PtdG24lUpxsTfYalHNjFHNBPUfk288AcDM8ecb748deHjJ/cZf+9HisRczjB344dLbH1y8/bt4B8GXVEezKKaGno5hdiSQnk/+zRFiu3sIdSYwu1sAqAxnCHW3EOpNM/3okaaGz062TP71IZI3bCC8oa2JWEkpmXnyWFN3grXaJBTeGG7SXdmZwMVei4cR6pVpyeJLF9e7+m2z/HUiVh4uPh4REpQp0i02kSdDmYCcClTWrXfTGuD6FgPjT2C7Zfpab8LQYg2NlCJUFLWZOOlaCJ3QkseS0idfmeDU+KPkyqPMX9uZ8adWNabJ7BFUdW2VqD81xEqgsKHlZmbKp8lbU7zVX6h38fbGwtL8lrZtlHLjlAvnnfJLuVHsjl2k2rdTyJ59K4b4toWUksm5o0v285rNDyCl/+5i5jJhTReQvsRoixHd2oGdLTXSfmZ3C+G+dNAjNF8h1JEAKetC9ebjuIUq0vPRWy5oVuvLRtpvvWDPlZvP79cb2F9BH09fBgURVxtBxGTt/mAWVYrk2CC200k/EWIck6/iEUSwQ4QRCByu/jVeCNe3ODv1PPnyKL2tN9IS3YihRVYs4Pd9D8spMFM4xfDsq1SslXVeuBjWuj9cQ8RKU0x86S1wS1ZRhYEn7YaaXxEailAbhl8Cpf5asGrxpYcnXRb+CoNtVMJ6CzEjzVzVwFDDDQGw61tLllqrIjAsrFsx4kt3UVVBUDCjoAqtobkKwojLT/pCgKaDrgtUtVEUM69PxvXAdSSue+V88XQdDENQt0UCwPfBccCxJd5VaA+mamCaAk2Fed2y74Prgm1LlpD4rNt5DT24dkVpvn7fD+694wR/LwchFMxwElUz8dzgu6npYYxQAsd+q5yN386Q5MuLOw8AS6b63j4QCFVD0TRUI4IeiaOFY6hmGEU3UZTzE4H0PXzXwbctPKuCUy3ilAv4ro303EUp3pXCzpRwC1UiG1oJ96WwJvJYkzms2SLRLR2YHXHs2SJusYZnOSBEo3XWQiimDopY7Ih+BWSnciU/wnWG/xalApHrQ6w8XIblSfrEVkzCnJMnmCMwXBYIwkQoU3gLndeb4UuX2eIZ8pVxktE+WuNbSIS7CRstqGpQBTzvpi7r/3O9GuXaLLnyGJniGUq16XX5zILnvliReP5iuEaIlWBX+/uoOHMMZl8CBN3x3WxK3cZg9iUmisdQhMrGlpuJGimOTj2KqUXpiG0nFe4nrAVCs4qTY7o0wHT5dOOmpCMb6EnsJW62E9ITbGy5hZ7E3sZv/+jUj8nXzkcahFBpjWykK7aDqNGKIlRsr8pcdZSR3CEc/3yIW1U0ehP7aItuIaTFsNwKk8XjTJVO4cmlfxyqCr39KvuuN9i6Q2PjZo3WVoVIVKCqAsuSlMs+05M+I0Mu5wZdzg66jA57VCpyzZOWEJBsUdh7vc4NN+ls3a7T0aEQjir4niSf9xkb8Thx1OHwGw4DJxzsVX5XhYCdezQ2bQ6+Xq4LR950mBw//zAIhQTbd2ncfJvB7n06Xd0q0agCSIpFyfiox8njDodfdzh+1KFWXftsLQTEE4Lde3V279PZvFULzhtXCIUEriOpViS5nM/4qMe5QZfB0y5D51yyGZ+LPcvmZk7Rv/19qJpBKRcQgkR6E/GWfs4df2S1o1z5SvwKO1EHWMV45nFVxsXK22BcrfFcBIpuYibbCaW7iLb3E0p3o0cTKJoR/KdqoKgIRXC+yVnglC99D+m5+K6Na1Wx87NUs+NUZsepZSewS7lgJbZC+LZLdThDdHsXejJM7uA5pOtTHpgivrsHgOLJCXzLxZ4p4harRDa2oZhaw31d6Crhja1Ix6M2cfn61iDqJK5YOm+18H2X2eJZirVZanYe5zLK/tcKKeW6EboiOU7JN1BQcHEaC36JZJpxZuUkNlc/3XkpOF6V2cIAmeIghhpG18L1QpVIUEksg4pC26tg2QVsr4LrVtc1gt2W3E7ISDAyvbijyEpxjRCrICKUDvczyMuoQicV7sfUYrRGNjFRPF5/rZeaW0LiE9LidMZ2UHOLTJVOoSo67dGt7Op4P0xLJktBNYfllpgpnaFsZ9mSbmG2cpZMZYh5hlJ1zosmBQobkvvZmr6LipsnXx3D9W3CepKonlr04XXGduD6FtnKCLnqGO3Rrezu+ABCKIwVjrCQBSkqbNuu8YnPRLjvfSG6elQ0TSz7bHBsSTbjMzri8tTjFo//qMrIkHdZz4pUWuEDHwrx4Cci7NitEY1dfELzveC8L71g8Td/VebYEWfF0SNFgY9/MsKv/Hog7q5VJX/yRwX+9q/LKArsu0Hnlz8f5c57TdKtykVDvr4vyed8Drxs85dfLXHkDeeyn5HpVoX3/UyIj38qwrYdGrH48l4xvi8pFiRTkx5vHrL58Q9rHHzZwrpgLspMHgMEHX03kmzdCkhq5SxDJ39MbnaxYPtiUM0wvXd9HDPZvuy20rWZev1JiqOnVnz81UI1wnTd8kEiHRtWtV9pbICpQ0/gu1duxZ/cfD3t190btOu5FKRPbvBNZg4/c8XGshSEqhNOd5HYsJt433bMRDtaONZoJbT8AUR92/PiYiMOkbZekluuR3oudilHZWaEwtBRSpPncEo5ll15+ZLSwBQ9n7gFr+ZQqTdWLp2aoOuj+7GzZarngp6ZtYk82ecHSN2xlcrQLLkDZ5ESWm7cSOvdO5h7ZZDq6OUTK2sqj54Mk7xxE17NqYfraVQIXm24vsWJiccWvHL1CbmPv64pSA+XpdaDzjVGqC6ElB6WW8JyS5RqV7f5dyLSRcRMM8LbnlhBoTZFKtyPoYYRQiVqpJkqnSJutKMpBppiEtaTTJeDvn752iSvj3+/KYI0Wx7khu4HaYtubhCrkj1LyZ4lYXaxqeUWCrVJJorHWepHkwh1sil1G5nqMMenH8f2zodJFaEtCg0KoXBs+jGKViBQnioNcHPvJ2mLbmGqdKqRsjRNePCTEb7wmzH6N6qrMn/TDUFnt0pnt8qNtxhIKfnWX5ZxVxml3L1X53f+XZw77jExzeXPr6iCtg6Vj348wi23m3zzv5T4b9+uUJZ2YbIAACAASURBVCyufrLRjYBUahp86OfC/M6X4vT2L38fFEWQSqs88OEwu/fq/PH/VuAnj9VWfe17rtP5H/9dnNvvMjFWcO0Lz59sESRbFLbv1Nh7vcFv/koGy2qOEEjfZXb8DTITh1G1QFjpOkHVzWrg2RZaKEq0c+Oy90ZKn8LoKYqjA6s+z0qhReIkN+7BTK6uxYei6cwee+GKESuhqEQ7NhDr3rI8OXYdsgNL9/u8ElBDUWLdW0hvv5lYz1ZUc+VakZVCCIHQdEIt7YRa2kltvQErP0P+3DGypw5Qy01fMopVGZrFaI1ROjmBNRuQGGsyj/QlWsykMhxUUEnXY/TbLwHQ86nb6H7wZqSUqCGd/OtDjP7di2tqjpx9YYD0Hdvo+8wddH10P77rUTo5weD//dgVsVBYGd7a6KaUHpP541TsgLAWa2srfhEITMIYhBAISuTw8OppNfFT0XFgPrW30m21NQrX4RoiVmVnDkWomFoMTTERKMxVRkiaXYS1BLoaRhEqJStYTUl8HN9CU8xA41QX6Lj111YPQcLswlDDjOUPN5EqYMl861xlhLJ9XuhmuUVqbhGjPlYIdEMf/liY/+GLcdo6zkdnpJRYFlTKPpYVRIiEAFUTmKYgHBYYJgu2h7msz6sv2asnVft0/uB/T7L3Oh1VPf8Fcx1JqSSp1SSeJ1FEcM5YTGk6d2eXwm/+Tpx4XOGvv16iXFrd5KMo0LdR4977Q3zxdxN09SiNXk6VsqRcljiORACaLojFBOFIc0Spt1/lt78UJ5vxOfjKyh/YPX0qv/cHCW682UDVFly7G5y7VpW4rkRKUFWBrkMoLAiFRdO9khIOvGSRzSyeiJS6Q7jv2bhLCK5XDOlTm5sk0b8TligjXgghBGaiFUXTrxiBCaU6Uc2VtYJYCD0cw0i04pTX3k9uKSi6gZlsXRFh8T2HWvbK67IUzSDet4P0zluJdW9FNcPrTqguBqGomC2ddNzQTnLTXrKnDjB35nXsYnbJNGh1OMPZP38Ca6bYaMJszRQZ+ounAIm1oA+gky0z/NfPMvv0CcJ9aRBQG8tRHpzGKzdHPfJvDHHu607T/peCNZHn9P/1CIl9fRjpKL7jUR3OoIUNJIEIfugbzwbi+gXX4bseE/94AOn5QaTrHYZCdYJCQ0t4+URPx6CHLXSJfkJEsahyWL5IhSLt9BIjyTmON7W8eSdiU9ddREPLZwEgIKKJSDeFyvjyG18C1wyxqrlFpAxSfGEtgePXKNrTSHzCRguGGsHxbSw3IDymGmisWsK9mGoMVdGDXmB6SxPZWSkECqYWxfVtLLe4/A4LxjwPiURKvyGmB+juVvnsF6JNpKpWk7z2isULz1qcOuEyPeVRrUhUFWJxhY5OhQ2bNHbu1tm5R6e3XyUSERx902Hg5Oomkp4+lS/+bpx91+soikBK8DzJ0FmXJx+tcfh1h/FRj3LZR9cF7Z0Ke/bp3Ht/iOv3G4TCAcFJJOHTn4syOeHxvW9XVpWSE0KwfafGv/n9OF09wb2Zy3q88qLN809bDJxwyOV8hICWlMLuvTrv+2CIm241CYXqK3Qh2LxV4xOfjnDqhEOxsLICgY9/KsyNtxgNkuT7ktFhr37tNmOjHoW8j+dBJCJIpRV6+zU2b1PZs89g23aNRItCtSJ54se1Ja+7a+Nt+J7N1MjByxYVz6OamUD6/vIpLurESjeuGLEKp7tQ9NUvUhQjRKilk/LE4BUYFSh6CCPRuqJtnXIBp7LyPoirhhAYsRTt++4htf2mIN0nrr5mSAgBQiWU6qTr5geI9+1g5shzFEZOIN3mOcMtVJn+cbO3l19zmH3q+JLH9qo2xaNjFI8uXVAwj8rgDJXBladthK6S2NFBYlsbWtQIjEq3ttJ5/w4A8ofHmPzxkcU7+pLs8wMrPs/bE2uLnAkEvWylV2yhRJ4KZRK0NIqxBIIusZFJOUyFlT3v3q5oTWwlYqZxVtSHUKBrq19MXohrhlhZbgnHt4jqKSJGippbpGxnsdwSMaMNXQlhu2Vsv4oqdLa23k17dCsz5TOMlA5huSVUxWR3+/sucwRBOUswQa1spSlXIBq96VaDbTv1Bqnyfcn3/6HC1/6syOy0v8SD2uMoABahsKCtXWHXHp1732vy3DMW1ir0lKGw4Bc/G+H2u00UJTi/60qeerzGV79S5PQpd5Fu6swAvPy8zcMPVfnM56J89guxBrlqScGv/ssor75kMXR2dQSivUOlvSMgC+cGXb7+/xR58tHaIoI0dNbjjdccfvJYjc//Roxf+OUoofr3XFUFd9xjsm2HzqEDy5OJVFrhfT8TbpAqKSXDQx7//ndzvP6ajXtRjmqjKMH+/RtV7rzHJJZQOHl86R3iLRso5cfWpXrJyk3ju3ZTa5qLwYinUfQQVNdfk6LoBmaqcwUEb4l9NYNQqiMQFq6RaC4FPRxDj6zMGdkuZPBq61v+34BQiPduo/PGDxDr3nxZ9+pKQNEMYj3bCKW6mD36PDOHn8Gz3xp37Ush3NvC5i/cTeH4BNWJ/KLfj73Otg0/TTAJ0yH6mGGs7rSeJCFuabxfpYyCgkn4HU+sAEamX2Vy7siCdmBLQxEKm7ruQVXWRo2uGWLlSzfQQoW60RWTmcogvvTI1yZpCfUg8ak4c3i+TdRopSO6jdnKWU7O/KRRgRc1Wht905aChEAUyuL1gMSnYudQhU7cbG/optaK628y0Bbc5Ykxj7/5qzIzU8s/hGvVILoyH2FZLW653eBjn4ig63UXXk/y5KM1/ug/5JmevPj5pYTJcZ+v/qciLSmFT3w60hQ1+vlfjPCVLxdXZEVwISbGPf6P/5DnuaetS+4/NenzF39eoqNL5YM/G25w3VRa4Y57jBURq95+la7u898Hx4F/+m6FAy8vv6/vQ2bWJzPr8/pBB03joilYxyrWCTlrlmg41SJ2MYsWWr6lhWqEMBNp7MLs2k66BLRQlHCq67LSWUIIQqlONCOMW1t/0memOpdtQA3BwsfKT18RUqHoJm2776T9+vegR5NXLe23Uggh0CNxOm/6AGZLBxOvPIJdXJn79FWDL7EyJYa/8yqVobV7B60vLvw8316+aToGGhoZOYmDtWj0Ph4gUbg2FgPrCUVomHocgQjsj6RHxcpQtVZSaCGwnCIRM7WmMVwzxAogX5tgU8stQY+vumgvX5ugL3kDtldmuhQI16X08fHQlRCGFsX2KhhqmP7kfkwtSnGJggfXr+H5Ni2hXmb0QVyvhhAqjl9D1v2p5qqjFK1pNqdux/EsitYUUkoURcNUIxStWTy5urRLS6o5LZDL+eTmVs9IVktiYnHBRz8epq39/PlHhz3+6mulS5KqhajV4G/+usy994fo7KprxoTg/gdCfOdbFcbHVheN8DzJ3/9NmReeuzSpmkd21ufhf6pyz30msXhwHboO23fqGCbYyxS2RCICc0Emy/Mko0OX501yKV3b9NghujfdRbJ1C5XiVNOqyPccfG/l3xm3VsbKZwi39S0v7tcMzJaOK1IZaMTS6LGWy97fbOlADUevCLEKpTpQ1OUjer5jU82uvxmwGorSfcsHSe+8FUUzrjlStRCKqpHath89Emf8pR9QmRl9q4fUQHUiT+6NUXb+2weoTRfxqs0R4fyRMaYeXdzS6EohHO+go/9mwvGOpnSulD5DRx+mWrq61WlrwXwF+8WIk0awMPF452nUEpEe9vR9BFUxyJQGGRh9gpq9Ur2npGbnm+Q8l4NriliV7QymFqfiZKm5wY2ougVAYqhRSnawMrfcItOlATpjO9nX+WEst4ShhrHcEvna4v5lAFWnwHR5gO74bsL6R3G8KiA4NftU47hVt8BA5lm2t76H3R3vx3bL+NJDU0xsr8qRqUfwVqlnKZeaGUQ6rdLWrjCXvbKCwY2bNW661WikAKWUPPe0xZmB1RGLobMuxw47dHSe14j19Krs2K2vilhJKZmc8Hj8R7VLpOAWY+CkQ2bWbxArIQStbQrRqIJtXfoeWlZg8jkPTRVs3qojlNpqrH+WRSzZSyTewbbrf55aJdvUuHhu+iQTQy+t+FjSdbAKM0FV13ICdk3DTLRyJVpThNt7V0ReLgbNCBNqaceam1rHUYFQNcxE23lH3UvAdyys3PpEnuehhqJ03/wztO66HaFq1zSpmocQCrHurfTc8THGXnqI6jVCrkIdcdru2YY1U8TOlPGd5vnEq149g05ND9O3432Eoq0U54bxvQUTh5R43tuLgNhYWFRpFz0UZBaF+vyJgkGITtGHjU2Vt8ba4krC0CKEjBY01cCoRVctRJ/IHkbwjiJWcwznXqPqnjdns9wSw7lDKEKlUu9K70mXwexL5GuTxM0OQDJTPkO2MkQy3IOuLBafSXzOZF+kaE0TNdoQCCyvjO1Vm7bKVIapOv9MOrKBiN6CQMHxquRq41j1SkHHqzJWOELemmzytpLSZ7o0gBCi7gAPp0+5+H5gDArQ0aXwqV+K8LU/KzGXXUpjtT7Yf7NBuu38l8OqwWuv2FQrqzuhbQXk5p73muj152woHIjRn3vq0pGcC3HquMPwudURu8yMT7HQzIIi0aB6cTnMTPnMzfkkW4L7oOnwoZ8L8cKzNd485Kybs7xrV8hOLr2yrlVX7/NTm5vGdx1UYxliRSCcVnQD31lHXxohiLT1IdTLTxMITSfc2kv+7BLi4zVANcMYsZYVERqnWgoMNNfx3B3X30fr7ttXpIG7GIKIpgw0Rb5fd1m38T0PpAdCRVE1FD3o/oCiIJTV2bRcCKEoxLo303Prhxl55rvYpavbuH7JMekqXsXm3DdeoDqWCwxDF0B6V69aTTMihGNtDB9/lPzs4KJqSilXN1nM2xmsFZdrh2BTY0IOsUnsYo+4FQcLDZ0esYUQYeKkGJInsLj2tHdrhaqsLYrsr4Pz/TVFrGyvzECm2cjP9S3Ozr28xLYVJorHmCg2P9CmSxevFpknRJeGpOLMUclffOKxvBKnM88usafPcP61ptcOvhKU6M8Lt1VV8MnPRGnvUPnetyscOmiv2r5gOSgK7L1eb/KryuX8VZOaeUyMeXguDWIlhGDDJhXdELjuysYuJRx+3Vm1VYTrBbYICxG0AVr+hzM54XH4kN1wgJ/XiP2vX27h775R5uknLCbGL89sdSFmJw4vv9EqUJubwncsVGPpZqMNCIERT6Ga4XUlVqoRxky2r/FBrgYaLUVdc6XkQmhmBD22Mv2DlZtet/siVC3QVO27Z0X6rovBs6rU8tPU5qapZSew8rM4lTy+4wTFMNIHoSCEgqKb6JE4ZrKdcFsv4XQXRqINRdMvT/umqMT7dtB1888w+sL315eMXwacXIXqeI7+T91C6czMeZPQOirDWXJvXL3omufaWJU5pH8586QgpMeIGGlCehJdC6OKBb3CLhOTuWNULqPKHWCSYXzp0SO2kKYTRah0yX4qlBmSJxnnndnDVFMv//cJYOoJVEWnYl2+JvGaIlbvRJw+5fLQdyt89tdimKHgR2aGBA98JMSd95ocPezwyENVXno+eMivR2+8eELQ26c20oAAnivp7lVXFOm5EG0d6qLFV2ubyqoCGpJVpyHn93MuIG8rLdz0PPjbb5S59U6Tzi6lIcDfsk3n9/99kk9/zuXZn1g8/qMaAyeddSe4lwu3UsAu5dCjyWW31WMpNDNad91eH+jRZN3OYA3ESgiMZBtaOLauflZGPI1mRpbdTkpJNTuJtx7kQSikt99M+w3vXZ7sLgHfc7Fy0+TPHSU/dBQrP4tnrbbiTaBF4oRSnbRsvo7Eht0Y8fSqCZZQVFI7bqaWn2bmzWfWlfSuFloshBo2EKpCYm/P4g0UcdWIlV0rUpobId2zj+nhg3hOjYXpdd9r7kE7D4EgHu6iN3U96egmwkYSRaxfirhYnbpsYuXjMckws3KCEBEUqeLjY1O95lrZrCcCH8zLv//d6X1EQq0cG/rB5Y/hsvd8FyuC68A3/7JMokXhwx8LE4ud1wrFE4Lb7zK45XaDkSGXAy/ZPP2kxdE3bWZn/MuqugOIxhRS6eYccU+fyp9+LXVZUhyhLCYyobBYicylAc+H2ZmrP4kffdPhK18u8K++GKev/zzZ1HTB1u0aW7YFVY5vHrJ55icWr7xgMTrsUaut/EYJoRJr6SWa6Ea5oEy3XJgkn1l5WxsAz7Go5aaIdm5cdlu1bpZZzVzaY2g1CLV0XLIqUUqJ79j1VNXFvwRmPI0eSawrsQqlu1fUFsZ3baz8zKr66F0M0a5NdN74/hVVai6E9H1q2QkyJ14hd+4wTinP5WvhJG6lQKlSoDR+htDRF2jbexepbTeu2t1dUTXa991LdXbsijr3L4fq6BzH/vCHF9/gKvZ4VDUDM5Kis3Mnbb3X41jlxvml9Dl35IdUS816PYFCb/oGNrffSVgPCj2uRc2di0OJK2PWey1CVf9/9t47SLLsvO783efTm/KuvZ2e7vEOAw8CGBCkQGpBE6QkMsRYrtduKILaf8SQIlYUd1dkSKHQSlwqyCVEip4gQYAECDODAQbAzAAY0z3Tvrqru3xlVWVW2ufv/vGyqrq6qjKzTHd1A30mJtpl5buZ+fLd877vfOeY264UCgSaFotyCXeAB8TqLmC+EPLv/02F0cs+n/65OAcOaqjaqvGlpsHBwzr7Dmh87JMxrlz0eOmrNi99xWZqIlgjwO4EpinW5QAKIbZWYWqDreamep7E24Ow+DCEL36+QWEu4Od+IcFTz5orOYHR/5DJCt77QZMnnzWYnozz+rddvvp3Nmff7KxNm+05woETLxCEHqaVwWmU0M0EgecwMfr1ra/Zd3FKhaZRaJs3WiiYub4tH6MV2mUDytBn8fL3yB1+BC2W3PRxim5i5vqoF8Z3bW2xXF/E9NsgsOu45Z3bC2hWgr7HPoKR7rw6JKVEBj6la28z+/ZLOMW53a0MNR36p177G+qFcXof+RBWrrclyb0deiJD75kPYJcKeHukt1IMFbMnhVOoELp7VzmDiDxVi+M0KuuHHSSSwF9f4elKHeJI7/sxtMQ9SajaoYsBaixh84PlF6YpxpqKVdzMEzNzlGtTeIFNKtaLqmzcLhRCJW7mOvKobLmGHf30Dxg0YZA3h6l4BRpB56Zpab2HpJZnzh7DlxuXWBcXQv7492u8/qrDT/50nA/9iEXfgIqmrd7lqGqUTffEMwYPP2rwyU/F+cJf1fnKF22mJzvXAmnaqh7qjmGL15EwiMxR9wKeC6++4nLlks/7P2TyqU/HOXVax7TESgVLiChG6NARhf0HNT7ycYtvvOTwl39a592zLm4LUpjvO0m5dJOZsVfZf+LjjF95CUVR6dv3FE5jG3eKUuIszUc6qzaRMkJRsbK9Wz9Gi+eLdw+1fEzgOpRvnic5cBDVarGpNEXwxcvf25W1KZrRseO6b1d3LtAWgtyRx0gNHemYtETVPIeFi68x++bX8O+AeesyQs9h8fL38esVBp7+UWLdQx1v8EIIEgOHyB46TeHcK7tS2dsqrP4Mh//HD3LtP3+T2ujeWhn4bp2p0W+y+YVt7bVLUy0OdD+Nqa/eWEgpCaRHwy3heNXIQ2mH1UDHvzPnj0CwTxzjpry0q8RKINC1OHEjj6JoNJwijlfe8H1QhMZuCPxvh6Zaa573QP/zdKUPc2Xya8yVLnJ0+EdIxno3MQsV6JpFoXRpZ2vY0U//gCFj9HEq+yHGqm9xvfoGnZTIBYIe6yD7EqcpewWqG9zZLMPz4NJ5n9/8tTKf/ZM6H/qoxQc+bHHspEYstnrhFkJgWVF48JFjaT7ycYvf/a0q33zJ6XiS7fbTdWkp5J233G23F2/HlUv+loXoewkpYX4u5LN/0uDlFx2eeY/Jx37U4pHHDbp7lDV6NFUV9PSp/P2fjvHcew0++6d1/vQP6izMb/zm6Uac0vwodn2RMHAJfJtKZY501yGy3UeoFG9ueb1OeZ7AbXSU1RdNBpq7IkbW42n0VK7lBu1VS9jFWdxqCSs/0PL5Yl2DuyZg12Ip9Hi6I/Jgl+Z2rK+ysr10n3p+S7YToe9SeOcVZt98kdC7CxNXMqQ8cQkZBoy8/6e2FJitaAb5Y09SvnkBp3T3iY2aMDDzCbylRvsH3zXcds0XCsnMIHZ9Ed9dJSDZ2BCZ2OoNSChDSvUJbs5/l3JjGi90kGGwY2IVbnEasVMIFHT0HWmRNkLC6uHwyKOk4wMIoWC7ZUZnXqZQXttyNrQED438GKae2tXjA8SMtf57haXLeH6dan0WAehaglJ1gkpjvcedEArd6SM7XsMDYnULGkGFBWecslfgTuoOPA8uX/C5fKHKn/x+nVOndT72oxZPv8dkYEhdcUoHMEzB408ZHDqS5Td+rczn/qLR9uYyCMC9rX1447rPP/lvF7F/8KZrt4yFQmQ8+rUvNThwSOPZ95p89BMxjh7XSCRXw5+FIhgc1vjl/znFkaM6v/4vl5gvrH/zPbeObkT6m8B3SaT7adTmURQVIbZXOnSrJbxaGSOVb/k4IQR6IoMWS+LuArEyMt0t42KklLjlBfxGlcb8JOmRE5vqGYQQGMksejK7K205PZFGi3d2IW4UJpE78NNQNIOuk89iZjsLb4VI4Lxw4TXm3nrp7pCqZUhJZeoq09/7O0be/2nUDvMdI4f8fvLHnmL69S9yt7VWftnGq9hoSRN3/t70U1IUjf5DzzE79jqVxRtA1C7KJUZWps+kDCmUr3Bx+svY3h3MpWwBFY0s3R37L2lo6GxjkqkN+rOn6HK6UJrxToaWYH/vsyzVJ3BvyepThEbS6t2xw3knKJQurVSglrVT0wtnKSytr0oJFFShYxmdRWZthgfE6hbU/RLnil/dtnfIdlAqhnzrGw6vfcdhZJ/Ke95n8bFPWpw6YxCLrbapsjmF/+mfprh62eedt1uLruyGpFYJ4ZY4F8uKdFe2/YOdZL4VOA5cuuBz+aLPX/1ZncefMvjoJ2I8916T7t7VKpauw0desJgY9/n3/6ayrmpYmr9KOrcfIRSKhSvsO/phugfOYCXyTFx9eVtrk75LozhDov9A28fqsRR6LLVz8iIUYvmB1nYCMsQpzxP6LvX5SaQMW17M9XgaI5nbFWJlprs6sjoIAx+7OLOj9la8Z4TswTNbagFWp64yd/ZlAncPKjBSsnT9HOmR4+SOPtF5S1BRyRx4mPnz37nrWitnrkLxjZsMvHCKmS+fxyvbawTroePjV+/c9JqiGpjxPK69BFJiWOl1+j1VM9HNtWReESpJa5VwO36VGwuv7xmpArCIcVI8iYbe4f4l0O7A9q+p6z2kDDXeMmrubkJKSal6k4a78RS1JGzG4OzsJuO+I1aaMMiZgyS1fOQTcgsq3iIF+zqaYjAYP86SO0fRnWb5TkwgGIgdQwiFmcbVlYzBLnOErNG/UhYt2GMseRs7NhtKnLw5SFzN4IUORXdqV4qpvgfXRwNu3qjxtS/bvPBjFj//iwkGhtQVofXAoMqnPh3jyiWvZRhztRqyuBhySK4WE5JJQVePsmk764cZUsJSSfL1rzp87zWXM48Z/PwvJnj+AyaqujxgIPnEj8f43J83uHZ1bQ+0OHeJ8uIYvmdTnLuIYaZIpPuZufE6xbnt9erDIMBemEZK2XaTFJqOkc5Tmx3b1rGWoaga8Z7WUTqh72E33czd8jyBa6O0ELALTcfK9VGdHt3ZlJdQsDrMLvR3aAyqaDrZQ2c6srtYRmDXmTv78q7aXmwVoe8yf+FVUsPH0Tus7AkhsDI9JAcPUbz8/Tu8wrUwupMkD3eTOtZH/qkDUejyLTrMhdfHmPizO7emdNdBep48wuhbnyXwXY4+/tMoqsGayp0QWPG1VRWBwFBXLT+q9jw1e/fzOrcGQUjIuLzSUaiyisYBcXLXV7FUmyIeqhhaEgTIMKBUn8BvEeslpYxanruk8xOKhkBseK0Ipc/VyRcJW3iVLZRHKWsbJ7h0ivuKWKlC51DqSXqtQ9T96AKW0rsx1TgLzsSK4NxQ4uxPPsp49R1K7qo7ukBhMH4cRWgU7BsrxEoTBnE1Q0xLkzX6ccPGhsTKUlMcSz9L3hzGCaK4m77Y4V3tgwc+zEytBjX/yq+m6e5Rm95NgiefMcl3qUy3iJOpViST4wFPPL26KWfzCvv2a1y+cB8Jo+4ypIRKWfKtlx3GRn3+xa9neO595hpi+/AZneuj/hqOEOUBRudS4DtMjX0LRdEjB9/tkgkZ4iwVkL6H0FtXaZZDj3cabaMYFla+v+VjbiVWfqOKVy2ityJWQkSiakVdE/WzVQihYOXbTz9KKSNiVds+wTEzPaRGTnRk67B8zPL4BapTo+x1WG+jMEF16irZw490PiWoKGQPPULp6lt31dcq9AIqF2eoXNw49qg2dmfJSr0yy+LEGK5dRjeSCKEwd/O7eLdoqVRVp+/gs2t/UIg1tioNb2llL9lLBPjMMt6RtYKKyiAHd30NhfJlKuXLdKUOoigaNXueqcWzBOHmlUeJZGrxLSqN3YmfGso/Qjq+gS/arRCA3Ph6uVTbuXXNfUWsskY/w/GHuF59g/HaO0hCeq2DnMx+kNnGVSbrF9nOhW3WHmXWvkaXOcSZ3Mc3eZRgOP4QXeY+rlffYLp+GRD0WPs5kn56Jy9rQ/g+fOVLDZ57r8mnfiq2SpCyCv0DrYlVGMJbb7h87JMW8Xj0c7GY4MlnDF75uv1AZ9UBJicCPvOfazz6hEEiGb2Hiio4eERDVddG+SQzQ+jWxhUCp16kXtleXp5bLeLbNYw2xArAyvVH5ckdVIXMTDd6rLW2wKtX8GpRy8O36zjl9oHRsa5BFM0g2AGxiqJsWuvNluGU5wnd7Z/k6f2nMLYQQB16DgsXXtsRcdwthL5L6fo50iMnOhp8gIj8xruGMNJdu56t2ArufJXxP727VbI1x2+UvQvbBQAAIABJREFUmJ94CwBNj1OvzLAweRbPra08RtFMMj2H1/6glAS3xJ6Eob/j1tFO4eOxwEzHxp9h879dX0foMF28xEzxHAiloyigMPSYLV1gsbo7TvDpWB/p+AAbTRwqQuPk/k+iCJVyfZpqYw7bXcJxK8384N3BfUWsLDWFIlSKzhS+jEqLJXcWL7SJa1kEYgdTGM1y5CY/byoxus0RKt48U/WLuGH0Icw0rtJt7SNntGHI24DrwPVrPp4HZlNnqGpgdmD+/L1XXaYmAo4cWzUk/cBHLP7mrxqca6PReoAIkxM+xWK4xhMsnhDrtNp9I0+S6V69+AqhoJtJZOgzfvXr2ydWlSJevYyRai/wNFJ5VMPahqP3KmK5flSrtau5szS3Qlpk6GMXZ5FhgFA3v5To8TRmuot6YftrM1I5tDZrW8ZOfLO0WJL0vhMtX8+tkFJSnb62q15dO0V99gbOUqGtH9mt0BNp4j3DzenAvSEJQhERQdmDwzuNEuOXXsTz1m6uMvQpzl7CtVfba5IQ21v9s65aKEIllHtHrB0aXJVnO368RHJFvk2DWvsHbwMSGeVedoBQ+gTh7pkc+qGDRG4o0ZGELJavk0sdoDd7kqHuxyKbjMCl4Rap1GepNGZWCNd2cV8RKzesIwmJ69lmq04S1zJoQscOKjsebW0FTTEx1SQV5wb+LSdBID3q/tIdIVaKAl3dCtotuj/PpSPTyskJn69/1Wb/QW1lynBwWOXnfjHBb/zrMgsbTLc9wFqk0sqKU/4ylkrrg7NvXPoyypXVr5JQNJKZIboGTq1MEm0Hy223ThzYNSuBnshsm1gJTcfKR9l+m0FKiVMqrLExaCxOIwMvYvybri2OmenZEfkwkjnUDqJskCGNha2l2d8KKz9ALD/QuRlo4FG+eYHQ3wP3203g2VXq85PEeka2IGJXSA4eoXTtXPR53iUIXSV9coD8k/sxe1IEtkf53SkWvzeGV7p7QwAy9HEb69vHMgyYn3x7rag+DKjYM/TLk5EfmNmNplr4Ldpd9yKW2PlAyW4gDP01FcCdwg/ciJxvcOpLGTK18DbTi+dQFQ1TT2EZGRJWN6l4P7254xzofw8zi+9y4eYXtr2G+4pYLbmzzNvjHEw+TkrrIpQBWXOAml9k3r5J21sdsf3E8WUxXChv9yaRm2qs0hnBoSMaY9d8KmXZsQfVMg4f1XjqOZNb97qFhYDZ6fZP5Drwpc83eN8HI5+sSIAt+MjHLUrFkN/9rSqFQtjx3aGqRtWaQ0c0picD5mbvbWKmqnD8IZ1qOWRmJsDd4jXPtOBjPxojmVo9X3xfMnrFX/c5+t76DcCxl0hmh8n2HKVS2i6hkB2TBNWMYSQy2IvbE12qutWWUIS+h1OeXyMydUoFAtdGNTZvOwlVx8z2bt/PSlEw0l0dVZH8Rm1HAvLU0NEthSx79QrV6WvbPt6dgPS9JuH1EVqndh+CWNcgqm7g3y1iJQR9Hz7O4I8/gl9z8EoNjFyc4U8/TvrUIGOf+Q5e6e66gguhNrV1t34P5JqsQElIsTaOFzQwtDhxM0suMcJ06d6KjdExSJLFIoZA3XDnW2Bmz53XAxnckYrVZhBCoAgNXY1hGWniVhdxqwvLyKCpFg2nRN3ZXj7jMu4rYuWGDWYaVziqP0dMS+OHLvP2TWYbo9SDW0/qqJx8u3hTFRqaYmxLbB5IHz/00JWo7Bs0NxchFLRN7PF7+1T++f+RpTAX8M2vO5w/53F91KNSlki5sRxGiGi8/6HTOv/d/5Li2InVQE8pJa++4lIqdkZqrlzy+cPP1PiVf55eIQjxhMJP/4MEQyMqf/IHdd55211Zz+3rUBTI5iIn8mMndJ5+j8HpRwx+9Z+VmJu9t+/OdB0+9d/EePo5k2+8ZPPGd11GL/vMzgT4QfO93+Q19/QqfPIn4vz9n4mvxABJGQ0EnD/ndSRjEoCiqChqZ+2rzWAvzkStthaVJIh8l/RUfts6Kz2ewki31jCFro1bXnvB8e0abqWIkdy8XSmEiGwcdIPA2XoVQlF1rGxv2+qLlBK3Vty227miGZG9xRaia+zFmV3NQtwtOEvzBJ6D0jGxAj2WRE9m8e070x66HWZXgv6PP8zs1y5Q+MYVgrqL0BSSh3vZ//PPkH/qALNfOX9X1gJgJbvpHjyNEc+tMc6UMmTyyss49dVzv9yYZbF2g770CTTFYjj/GEuNaerOvVEF0tA5Is7QRTSMspmeqi6re0SslnMYJWHo7WrFKghcNqsYCKEw1PUYXZnDxMw8Uob4gY3tLjFXukilPoPtlvH8nb0n9xWxUoXOSOIURWeSi+VXNp3E8KVLIH2SemTJsKzHSus9JLQsFW/rJ78bNqj6iyvPEZmIgqUkyRobTysJAfluhZMP6zz/AZOlUsjMVMD1UZ/RKz7TUwGVsqTRCNF1QTarMLJf5dQZg1NndLp7lDWkauyaz+f+vN4yXmXN++DD33yuQf+Ayj/+75OYVqQPsizBhz8W46lnTW7e8Ll2xWdyIqBek+h6VJnK5RUGhzW6exR6elXSmdX4l93MHLxziCJqjp3UOXpC4+d+QTI3GzI57nP1ss/YNZ9SMaRWjUhlPCHoH1Q5flLj4UcMDhzSMIzVi6vrwmf/pM7EzfWkPJ0/gGndMpovBFaii1zvMaaufWtHr8KtlvAaVYw2o//Lk4FC0bbVyol1DbasOgH4Th2nsva7E7oN7NIcyYGDtKoGW7leVCO2PWKl6Vi5zmJ73PIC/jbboUYqh5nu7jz3rdl23M5rutNwK4uRE3+Lic3boZpxzHQ3jfndC/RuBaMrgWJqFL5+Gbe4+pmV3p4g98R+Usd67xqxUvUYI8d/hFiyB9epYMVz1JamSWQGKS+MEdxm+BpKj2tz3yJp9pAwu8jFh3lo8AUuz7xIpTFzR2UpncAiQRf9THODOTm+KbHaq2qVH7pMF9/B0GLYbhk/2L2JKj90Nr25VITKUM/jxM08c6WLFCs3qDRmqTXmCXdxsvO+IlYSiRc69MQOEtPShESlBzuoMWNfpehMIQlxgjqLzgQD8WOczLyfJW8OQ7HIm0Nr9FEQTQkktCyaMMgYfZEjrNZFlzmMF7o4QRUnrBNIj8n6eU5m3s9D2Q8y07gKSHLGIKq4zftkA6iqIN+lku9Seeh0560GiEjV9FTAb/7rCqNXtiaQbNQlv/Ofqni+5B/+4wTZnLISPpzOKDx8xuDhM1tbz/0GIQTxhODAIYUDhzSe/8DWfr5eC/n8Xzb449+vbfh97R44vUa8DpHtwtzEm8xPn9vBysGvl/GqRYxEmnZtbCvXh6JqBFslVkIQ6xpq2wJzlgrrSEToezilOcIgQGmps0pg5XpxK1svsevxdEs3+GXI0KexML3tyUgz29vRcZYReC6NxRn22mJhIwROnWCLk5GKbjYHJXZm29EpZNOzSrHWVtWEpqBaGqF396wfdCOOlchz7eznUFSN7sEzXH/nC6Ty++g/8MyG1hsVe5bzk3/Lkf4Pko0P05U8wGP7P81c+QrzlavUnEX80CEIvebU4Nbf01YDVa3g41Kngi1r1KhEe+U9BD+wGZ156Y49dyB9CMW6qcQg9Lk6+TVyqYNk4oPs632aUIb4QYNKfYal2iR1ewHbq+yI7N1HxEqQ0XtRhU7NL+KGDSQSBYWM0Ue3tY93Sy+y4EwgCblW/T4hPlljgJTejR1UuVE9R0rvIq6lV9xpY2qaU9kPooooN8kLG+TNIXLmAKEMmaxf4GYtmrZYsMe5yCsMxU8wGDuOGzaYs68z27jGUOLkursC15EUF8OVLLqtBqBLKWnUJWff9Pi9367y7W9ur/3WaEg+89s1blzz+dl/lODUaZ1YfGMDtVZrcWyYmQk6bkXuJaSUFIshjUaIZSlbfu8h0lRNTgT85Z/U+fM/qlOtbHyBG7v4dyi3XHglIMMQGfo7TkkPXBunvEC8d3/b12Aks6hmbMvO36oRiypCbQ5gL85saClgl+YIXQcltvnlRDFMrFw/5ZsXt7Q2ACPdjaK3H4WVgU9jmxozhNLMNezQ+4nI2sAt77Ux5MaQYUiwxZaeEAItlkKoO/Mc6xTObAVvqcHIp59g5svv4hbrKIZK7on9JI/2cfO/vnbH17CC5rnvew10Yqi6hQDq5RlUPYZhZfCc1RazIjRieqTJqdnzZGL9IAxMPcVI/jEGs6dx/Sq2X8XzG9smSGPzr1JuzGz552waTMrrDIvD5OjFobGhI/uUHOvIUPR+QqUxy4WJv0Wg4Kxzw5cslK+xWBlDEZF4PWF1kYz1kor30505hqoaTC+8zejU17e9hvuGWFlqgiOpp7HDGu8uvrSmDZjU8zySe4GcMcRis2plBxWulF9FFQYCQUiAFzosODdRhIrXnOCo+yXeXPjSpnvKrRWukIA5+zpFZxJFaEhCvNBBIFhwbuKGaxnu5ETAr/+LJT78cYvTj+jsO6ARj0cickWN9DzRcaM7xDCEMIg29FpNcv4dj2+8aPONFx2mJ4MdBSjbtuTLX7R556zH+z5o8oEfsXjoYZ14IlqPqq7uq2EY6ZP9ADxPUpgNuHLR5603XF59xeH6aOuLrpRRrM5SaXXBtWq4ZfE+RJeienXtc1UrYdv3wnHgjz5TY2425NnnDY4e18nmFXQ9eq2KEv1/65rDMDJodV3J1ETAq992+PLfNLj4rtfS+yue6sOpL67xv4kgsOJdmPEcdm0Bp1Fi6xdXuaqzaiPeVg0rio/ZYlVIsxKYmdYtMBmG2KXZDcXnztICgdtAiyU2/XmhaJGAXdW33Ko0M10oHXh5+U5j29E5QomI1VYQOA28+r25KUkZbitaR48lm+3kO0+svHKDib/4PiM/8xQn/vcXCB0foSlIL2DmK+cpnbs7LUmIKq+BZ2M1v6uGmSKZGyEMfTQ9xu3f22x8mFPDn0RTTFRFQ2mmgAgECIGmGmhqnrjZmffaZpguvQtsnVjpGPSKIRKkMbHw2Pi7Oc/0Dxyxcv0as6XNW8iK0FCaE4ExI4NppNG1WNP0VSJQVj7P7eI+IlYp4lqGmcoVnHDtBmYHNULpI1CaXlYRAulHJcFb4Et3zXdEEuKEWxG7SjzpgHRu+RtwwvW9at+H773m8vYbLrl8pFXad1BleCTSLiVTCpYVkawgiMjHQiFk/GbA5YseUxNRdWi3vOfCACZuBvzx79f50hdsRvapHDups++ASle3ihUTBIGk0ZBUliRTkz43rgeM3/BZXFjVI7U9Tgj/9TM1/vbzqxf2MIDxG1u/WHsu/MavlYknb9E7OZLCXHuWNjMd8oe/V+Pzn63T3aMyNKwysl+lb0Alm1OIxQW6Fp0vji0pFUOmJgOuXfW5cc1ndjbA60DP1r/vKWrlGTyngpQh5cUbeG6VRLqfgw99Et1I4PsNrr3zBWrlrVsBNBamI0LThlgpuhkJ0KdHt/T8ZroLLdY6AiVwbdzKxllyvl3FrRYxM92b/vxydIpmxvDqnRMroWqYqa624n1o6qvs7WlGFM1oKcDfCH6jck+Ygm4IuTzJtjVosSRCVeEuDQYW3xinPlEkdbQvsluoO1Quz1KfKCHvYivQc2sUZy8hFBWnXqJRK3Dk8Z8GoFGZxWmsHVDQVJO40bmJ7N2GSYwMXUxwlRl5k5Dbp9kjeNw7NiF3A0KoHBx4L/n0IUwtAQj8wMHxKlTtAjOL71K353fkYQX3EbHyQhtfeuTNYead8ahSJASGEmM4fhJNMVnyZu65XjKA58HcbMjcbMi75/benFNKKC6GFBdDzr51Z9ZTmA0p7IIlg5QwMb79z1RKKC9Jykv+uoy/3YJuJhk5+iFcp4IQgkZ1nqtnP0u25xiB73Dt3S/Qt+8p+kaeYOxiYSX+plM45XkCu4aqt06jVzQDM90VBcluoQUZ793Xthq2TJ42Qui5NBamSQ0dbfkcZqYHLZ7Cq3ceVqsaFkabahrc4rG1TSG5Hk937FR+K2I9w5FI/B6Dohmd+X7d/nO62XkUzm5ASpzZCs7s3lZNZOgzc/07K1qomxe+TKlwFUXRqCzewF9Xjb634WJTYh5H2rjY+HeLKd/jEAg01aRUuUG5Pk3dKeJ6NTy/tqvRdPcNsar7S4xV3+Rg8jEe7/okfuhGb5JiIIHRyneZd+4d9+N7FaoZZ/DxF9pOmckwpHTjHRZH72DkhFDIDJ8gd/BMRIAmLlC68S6yRUDmvQgZhkyPvRplBAqVw6d/Ioq5MeI49hL1ygzzk2+x78THogzBLRKrwK7jLM1jpFq3FYQQmOluVN3oWLgsFJV4z0jbipBXXcLfpO0lAw97cRoZhi01SophEssPbGnqLJpU62r7OBn42KW5bZ87WjyFanQQaXALUkNH25LJ+w1C1bakvdwqut5zGDWmM/fSJbSESXwkR+XSDDLY+wEAoagYZgpF1RGAXYkmvxXVQFHXfm9rzjxXZ1++42uqbdu+QeLjsV8cZ4D9eLhIwnU1qzF5gQob3zD9ICKUPpfG/+6OH+e+IVaSkKn6JcruHCmjG02YSELc0KbqLVD3lzYU5z3AWiiqRmrwKLFs6/H1MAywy4U7uhYr28vws5/CbBKG1OARfLtGZfrqHT3ubiPwbRq1AoFnEwCeW0UzEmvIShB4USVgG5tW4DZwygskpWy76RnpLhTd7JhYabEkRjrfdll2caaluadTbuqsrM11VoqqEesegsvf62ht0HRcb2MDARB4zko49HagmfEtGYP+oEJRtW2do50ifXIAoSnMfe0i8eEc+372KS78n18iqO9tS0rVTAaPfoBM9+FIa7PGH1Ry9c0/p15e1TrVnAWuzX37jq9ru3uaioaKRpUSqy9GrJsrvnOf9A837htiBdFJVvEXqPj3hgnb/QhJNM0UeG60SQux8itsbVJwp4jl+jFT+ZXWgx5LkejZd98RK7u+SL7vJGHgoSgaqewImh7Hiudo1BZQNRPdTEQTgtuYEpRhgLNU6MhJW09kUM14x6aVeiLT1Ba1EK5L2Xbazq0s4tu1lsQKoUReW1sQsFvZXhSt/WUqcBrrJvQUFATrK2hygwBa1Yx1nA/4gwyh3FlihQAtpqNYOkJXUEwdoSoIdf0xpQTCu1PJ0s0kud5jzI69TmXx5lpCI6Mw9XXru4dv5G3qXJJvtn1c8KBFeEfw4EryQ4bArjH+7c+iJzJoZhzNjKM2f82MnNySj8/OIBCKdts9lLgvN7fC5FvsO/5R9h3/KCBZnL2EZ5fx3TqaEefgQz+KleimWprYchtwGXZxjtB32zppK3qks+o02sbK9aG00W7J0G9bDfLqFbzaElaLSqgQAiOVR48lN9Vr3Q4z1xtt9m1wu3A9SYYhcZgEqeZQy2p82Jyc5CaXb1mYgmrG7+pNxT2LO/wW1G8ssO9nn+bQLz2P0FSsvjRDP/Eoobu+hVu7Ns/id8fu7IKa8N065YWxqMosxAaEbu9blVuBROL/gAjTFaGjKtrKDXgog6Y32L2np17G/beLPcCOIMOAWuEGrOvyCY6+8Mt3kVhJnPI8vlNf2dRCz6a2g6DevYJdX+Tq2b/EsFJIGeI2llb8q2LJXvpGnqC8cJ2ZG68RblMD5CzNETj11hUhmvEvuT6Wxt7p4FkFib4DbQmFVyvj1VoLzmUY0JifbKs5MpI59GSmI2Kl6GbkhN6Bt1RjcXpNEPKIOIqGzoQcJWDte36727RQFFSjNbl8gN3B/LdH0eIm+WcOYvam0HMxup45uKHGSijirhGrMAyQMmTk+Edw7KW1E5Uy5Nrbf0W9MntX1rJbsIgzwAGyohuFjTWUV+XZeyaMGUBRdOJGllRsgEx8kLiZR1cjK4TliX8pA0LpY7sVqvYsS/UpavY8jlfec8f7ZTwgVg+wZ6jPjzP1xpfIH3oMGYYsXnvzvmsDLiMMXOza+gtUozrH2IUv7vj5faeBszSPmelp+TihapHtQQeTgYputPVuklLiVov4jTZTWzKkXphAttGBCVUj1jVEbWas9fMR+WsZyfYj7WHgYxdn11gf1GQZTRhUKBHgrbncrpscFgJFe0Cs7gaCmsvk595i8q/fIn1ygH0//wwXf/2L+LWNqit3b5M0rBTp7kPcuPB3VIvja419JWtyAu8HqGgcEg/TwyANakhC4qSoUMTAwsBilnEc7o04Jk21yCcPMJB7mExiZMUKAaJfbleHSSnJxKGPk0gZUHcWWayOMbn4FtXG3J63aR8QqwfYM8gwYP7SayyOvglIQt9F7sQF9QcYoedil+ZIjZxoTVyEQE9mUQ2LoE1mnpHKoyfaVCiljNqQHVgKOEuFSMDeZsw/3jPc9rkgIlZt10ck7r/dGLRMkaM8Ql704LI27X5BzjLFtZU/i/u0BX1fQ4JfdXBmy8gg3HYM0W7Bd+tUFm+gqgZCKFFqwvI/yu1QPIHaNKJUhIaq6NHzypAg9AilTyh9gtDnThBIE4sMeWa4yZi8QJIsh8Up3pWvo6IyIo4REuCwexl920Xc7OJg7/P0ZI6iq5215G99jBAaCauHuNVFV+oQNwuvM116Z1fzB7eKB1eTB9hTyDDYlkP0Dxtk4OGUCpGlQZsUbCPRjLZpQ6zMdBea1TqkN2rxTXS0Rq+2hFdbQjViLS+OVq4PRTPWtO42gpHp7mhSL7DrOLe5zfeJYTwc5uXUOg+fdU7Tgi1F2TzA7qAxVWLs918lsPdeQK2oOrF4F/m+E/Tue6LpZxVBypDRt/5izVTgps8jNBJmnkxskFSsn4TZhaklURUNEEgkYejj+FVqzgIVe5ZyfZqqM08od89mRkVHQWVBzuDQwCIeHZsAhwbT8jrHxRPESVFjZ2aYO0HC7OLE8CfIJfejiLXXtdXPQK5r8UUVrNVhKyEEApWE1c2RgQ9hGmluzH0HL9ibveUBsXqAB7hP4JTnCdwGSqw1GdLjaTQr0TLeRSgqsa7BthN3oe/SWOjMLd5rVHEri1i5/s2PKwR6PIOezOK0EcTH8gMdTai5lSKBvTY9oSFrODSYYwL/tlbg+grB3Z2GfYAI0g/xittzyt9teG6NsXf/dlOCvVGbfy0EKauH4fzjdKcOY2kplBbecEm66UoeIJQBjldlsTbG+MIblBszu9TGkiv/wWr7W8fEw8WhgYKCibVnxMrQkhwZ+DD55ME13z8pJa5fo+EWsb0KTjMQOQx9EAJVMTDUGKaRIWaksYxslPXbfA5di7Gv+ymCwGWs8J09Ebk/IFYP8AD3CZylyIFdb0OshKZjZnqoz91s+Zh4zwjtxsCcpfmOrRsindUk6ZETIDbfVLR4CjPd1YZYCWL5/rbrA6jPTxDe5rElhGCAg/QyjM/adktBTjHOlVvWLbccli2lRAb+fWdm2w6h59y1tpwaN0ge7aV2tYBf21v3ehkGNGoFYoluEtkhAs+mOHcJVTVQVKPlNK+mWAzlzrC/+yksPbMlkq4IlZiRYch4hO7kIcYX32B84Q3cYGeE08cjwCdBigWmcXEAsRLIHCOFRusJ4zsJIRQG86fpSR9deb+kDKnaBaaL55gvj2K7Jfxw8/NCoGDoCRJmN73Zk/RmjmPqyRV39ZHuJynVJyhWx+7Sq1rFPU2sFN3EiEcO4VKGuLWltf43QqDHUsTyg8S7hzHiGYSqIcMAr1HBWSrQKM7g1krNqIutXTCEoqLH08TyA8Tyg+jxNIpqIGWA36hiF2eoL07hVott2xptX6umo8VSmKkurEzPijGiUDVkELXL3FqJRnEaZ2ke367cN3okRbfQ46kN7OnWQhIJpXc3f01EIcPpLqxsH2aqC82MRV5KYUjou/iNCk5lEbs8h1evEDj1lmaYAJqVXNESeY3yiiGnnsiS3fcQsfwgMgxpFGcoT17Cray949XjaTIjJ4l1DSEQuNUi5ekrNBamNj22b9dwKotYub7Wr1iIto/RrARmtrUQXkqJXZrDtzuP86gXxpsO7JsTK6GoWPkByuOXNhXYq2YMvQPhupRhVFG77btQlAXqbJwB2mD965HB1r9LCxdf63D68v5B6Ltb+rx3gthghiP/wwe4+H9/Gf/anTUjbg9Bvv8hho99GMNKUSvPUCpcJZEZonffE4y9+7cbxtoYapxDve9lKPdIU0e1XmQdyqBJ3CPDD0UoCKGue6yhJTnY8x5iRpYrMy/j+NuP+XFoUGYBU8RQpIqLTZkFDogT9DCIRZwAf9PvyJ2Gpafpz55CNG/AwjBgtnSea7PfbLrNt9+rJSFOs6JVqt1krnSBY0MfJR2LKuamnmQwd4ZS9eZdF7Pf08Qq2XeIA+//GQB8p86NV/6M2ux1IIq6yB04Tf7w48Tygyia0SzjRo41MgyRgY/vNihPXmbqe3+Lb3d4EgmBlekld/ARMiMPYaa7UTT9lueP7nBk4OM1KlSmrrA4+ga1ws22G/La4yjosSTJ/kOkBo+S6NkXkUNNj8rITdPO5gGRYUDou9hLBZbGL7B47a11G/a9iPTQUYaf+nEUvbVeJvQcRr/6ezSKnXkwtYNmJcjuf5jcwUeI5QaiHDRFXfM5RtWKICKvno29VKBeGGdp8hK12esbfp5CUek58Rw9Dz0PwPSbX2b+0mvEcgMMPvECyf7DTb8pSeh7NBafYOL1z1ObGwMglh9k8IkXSA0cWfGlkkFA94lnmT37deavfHdDA00Z+NiL02T2nWz9woWC1WZ60Ex3t82SkxtM27VD5CdVaznNJ4Qg3j2EoqqE/sYXPC2WQrOSbe/+A8fGrayf2CqziIhy6rm96nX7RVZKSRhs/cbIKS9Qmbjc/oEPsCGEGrXdvPLeayx1M0HvyBMUZy/i1Et0D58BwLXLxFK9GFZqHbFShMb+nmfWkCopJX7o0HBLVO0CdbeI69ci3yUkgmYrS0sQN3OkrB5iehZVMRBCoAqd/sxD+KHLlZmvE4Tbu2EPCRmTF5tWuNE1bFxeQRXeTOG0AAAgAElEQVQaKXJ4OEzIUewNbjLuBpJWH5aRXXnPSrWbjM6+TN3Z3vRlKAMWq2NcnX6Jh/d9Cl2NAYJ0fBBDT+B4dzeLsi2xEkJYwDcAs/n4P5dS/gshxEHgj4Eu4PvAP5RSukIIE/gvwBPAAvAzUsqx7SxOqCp6LNX8vYYRT1MjuugOPPpR8ocf20QoK6Ivraqh6CZ6LNV5qV8I0kMnGHjso8S7hqKIhw3XpoGqoRoWZqqL1NAxZs++yMLV73e8EaUGDtP/yEeIdw2jGlbrTUSoCEVF0QySZoJ41zCpgSOMv/Y57GJ7UeVeQlF1NCvRNostUHXYJRGxle1j4NEfITPyUOvjimj7RdWjwN9EllT/YVIDhxl98b/g1Uob/tjyeQUQyw9hJLL0P/ojpIeO36LTEKi6SaJ3P4OPf5yxl/8QoagMPv4xMsMn1+g5hKZgprrof+yjuPUllm6e5/a7NhmGNBZn2loaAOjJDIphEW4SbWNmutt+HqHndGw0ugzfjnRWeqJ1S8TM9qLoFqG/cYtFjyfRrDbET0r8RmXDVqWGTh8jpER2jYePBIpylhluaZPKkNDb2gYmhIiE9VsMvL5T0BM6se44lfEy8i65le8UbqmOV2pgdidx5/emcrIMVbPQzSTzk2cxrNTK3we+jRAqirq+bZZPHmAk9xiqEv1bKANKtXEmim9Tqk/g+nWC0GPj6kuUc2tocbKJEYZyZ8jGh6NbAUVjMHuaYvUGs+VL235Nt1dm61S5LN9CQyckbLYH9wZxK4+qRDfaUoZMFc9Sd3aaWShZrFynWL1Bbyaanrb0FIYWv/eIFeAAH5ZSVoUQOvCKEOKLwD8F/q2U8o+FEL8F/BLwn5q/FqWUR4QQPwv8X8DP7HShQolIlqKbDD7xAl1HnlwhPTIMCTwnussXCqpuNkW5AhkG1As3205IRQcRZPedYuTZn1yzMUgp8Z0avl2LYkUUDc1KoFmJaBpBiTbFoad+HKGozF9+vSNyJYRCLNePZq5moS1rN3y7Gr2mMECoGpq5erzId0cnNXiU4ac+yfWX/6iz17dH8BpVavPjaFYCRdVRNANF06Ow0zsQ+mokcww9/WNkhk6skJeoKuHhN6oEng1SIhQV1bDQzMRaR3MhsJcKePXWppjLiOX6yB1+jPTQMULfwa2VUHULPZ5BKApCCJL9h8jsP41mxUkPnWDZIDUM/GiKzzABgRFP0X38Gaqz1zf4TCXu0jyBa685Z26HEGKl4uNuQKyEqmNle9sGL/t2bcv5e75dx14qkOg/2PJxWiyJme7axB9LYKS6EBtsZrfDrZbwNniOXkYYEUepUCJNPmqLEAckU7cZhEoZNqUCW4OqmwhF2VYbcbfR+2g/Bz9xhFd/7Zv4jftD9+Uu1ll8/TpDf+8R5tIWznxtjb7Lq9h3jXBJGSJlgHrbOadbKSBcJ/VQFYP9XU+ha9H3MAh9pkrnGJ39Bo7fyZqjypbvOtTdIguV6xzpex8D2dOoQkNTTEa6nmShOtZSZ9Q5BAKBj7duSnYvYKjxlSlA169Rrk+zG7YTofQp1m7SnT6KKjRU1VwhcHcTbYmVjGYel88Uvfm/BD4M/Fzz7z8D/EsiYvWp5u8B/hz4D0IIIeXOFJGKomIkc/ScfJ6uI08ihIJTXqA8eYny1BXcWgnpe9HUgBkjlhsgPXgMPZGmMnOt/QGIWo/DT/+9lTZGpJGZZuHKd6nOXMNbIVYqWixJauAI3Uefwsr2IRQFzYwx8NjH8O0axbGzbUWgtcINypOXyR9+DK9epr4wRWX6Ko2FSbx6mcCzm+P1GrqVIDV4jJ6Tz2Ek8xGhE4LU4DGy+x9m4cp399wLZjNUpq9QK9yIyJRmrJArK9vH8DM/jt5m5H9LEIKuY0+THjy2Qqq8RpXFa2+ydPNd3GqRwHNAhgglqjjqsRTx7mGS/YeJ5fpRdZPFa292XImI5QcxU100FqeYfvMrNIrTaGaCvtMfJHfoMRRVRVE1eh96HlU3CX2XmbMvUho7Rxj4pAYOMfTkJ1cy++Jdw1iZHmpzN9Ydy60W8euVlsQKIuKix5Lr8vMAVMPEyve1JbSNxZmtE3YZYi9MIX0P0aL1q5lxzGwPtdmxdf8mFAUr19sB4ZYb6qsAcqKbGXmDSa5xXDzONXkeH4/D4hQGt5mBShndNIVBW7K55jXEkghF3WVN4Nah6Ao9p3uJ9yTubM7fLiM2mKX7fUex+tJkH98XBTHfcg0rfPMqY79354OOATynQqU4wfDxD1NbmkbVLPL9J+kZeZx6pYB9W1ZgJjZINj4ERKSsUL7MlZmXtj3e7/gVLs+8hKZa9KWPI4RCJjZAOjbAYm1sm69KkCJLnxhutv9cRuU5GtRIkEZDvydc14PQxdtFz6moOhVdE/bq29CRxkpECrPvA0eA/wcYBUpSrhhvTABDzd8PAeMAUkpfCLFE1C6cv+05fxn45Y5XKhTSIw+tRE8sXn+LuXdejlojG0zmVKevsXD5NfREFre6cTvnVujxDP1nPoSRygORhqp44x2m3/g77KW5daTFqy/RWJyiOjPK0JOfJDUYTTdoVpLeU++nPj+B00b/FLg28xe/Q6M4TWXyCvbSbBSlsAFB8mol6ovTNBanGHnuJzDTPVE7QlXJ7nuY4vWzhN7em71tCCkJPWedyWQYeLu+KelWkvTA0ZUNMgx8Zt7+KvOXXt2w7eTVl7BLs1RmRilc+BZmugczld+Q1GwGRTMIXJvpN79MefJy83nLzJ77OvHuYWJN+wEr24cMAwrnX2Hu/Csrr7009g7x7hF6T70/0lkYFla2f8M1RJl8JcxsT0vioWgGeiKz4b9pVgIj3d3yNUkpsRenCdyt3y03FqYIfLelpk6oWnRDomrrzwGhYGY2zxy8ZZE05ic3fn4UfLyVsGUNjQZVGtRJixwFufbnArtG6LmobQjrrdDjqUgS0IF56jISA0kGnx0mczC6eSteWWTy2+PYC43lhXPgY4fpOtHN1b++xNL11WtXzyN97P/IQaZfn2TyW+PoCYN9H9xP7mgXQ8+PoMV1nv6V96zo1grnZhn961UNWP9Tg/Sc7uXKX14ktS/DwNODmFkLp+Rw42vXWLq2eizV0uh7rJ++x/rREwb1+TrTr02weGkBGUhUQ+XAxw/jlh0Q0PtIP4Vzs8y+McPIB/aTOZRj+tUJZr47tamOzpmvMPaZ72z6XrkLd689GAYeM9e+Rf+h5+keOoOqWew7+XHKi2NMXXl5jQZPCJVcYmSlEuL6dSaKb+7YM8kLGkwsvkkuPoKpJ1EVg2x8mGJte+Lrbvo5LE6j02y5EaKiAoIsPQyJQ7wtv7knJqFe0EDKACGiDEBF7J6PXFQJi66Nfug027F3Fx0RKxkZQTwqhMgCfwmc2OmBpZS/Dfw2gBCibalFCNEU5EpKY+eY/O7fbKp/aR6B0PdwljqZNhFk9j1Esm/ZT0NSX5hk5q2vYpda5ENJSX1+ktlzX8fK9mE024ex/ADpkZMUzr/S9sjVuRudi95lSHnqCoujb9J/5sMITQei90W3Ejj3KrG6i1CNaJpsmXR4jTJLExc31fKsQC4LzadoLHbm27QMIQT1+XHqt23yTnWRWuHmCrESQuDZVYrXz64hE2HgUZu7iTwZtX0VVcfYhBQhQ+ziDMk2mXxCCIx0V1TBuI2om5keVL0DfVVxdlv6IbcSVdVa2UIsTy4qmkFwG7FSdL2jKJvAczb9fjeoYYk4QgocGnQzgI9HjATuBhuJb9cIPGdLxEpLZFBU7faAnE2ROZTjyf/1GayuGLXpCgjB4HPDDDw9xJv/8bvUpqsgoTpZ4cwvPYae0HnzP34Pp2QT645z5pceQzXUiCxJ0CyN1EgGPaGjaErkEB7K1QDh2z737KEcBz9+BN8J2PfB/QROAAKsfIz5d+dWiJWe0DnxM6c4+MIRarNVvJpH/mQ3Bz56iLf/3+8z/o0bKJpC3+MDJAdT1As1UsNpht47wtR3JkgNp4l3x+l7tJ9vF75B6erGguSg5lJ6697JBnUaJcYvfoW5G6+jmwl818ZplAj8teeLIhSS1uqNTWT0uTtTjVW7QM1ZiGwDhGgeR9myHYiOwbA4govDFfk2OgZHxOnmv0rqVNAxsEjsCbGqOwv4oYuhaKiKga7GaNC+ANIJYkY2yhWUEser4vl3XyazpalAKWVJCPES8ByQFUJozarVMLC8q0wCI8CEEEIDMrB79UavVmbm7RfbkKqtQbMSZPedQtGjalgYBCxc/T6NjkThkurcGNXZ6+QPPQpE1YL04FEWr36/vau4DLfWwZMh5cnLdJ94DqOpC1J0Ey2eblsh+6FAU/O2glBu2CraTUgpaRRnIu3WLQh9F7s0u0Zsbi8VcKrrNxqvUSH03Ug3KERk5bCJMLo+3xnxMzPdLE/J3gor27tWU7YBfLve4U3JegRuA7s0R6yrtQ/VsoD+9najHkt3RHC86tKG+iqAeTlNhjwSybyc4oR4gn72ExJwRZ5d/1z1VcuMTqFZCfR4uiOfLy2mceofnCbWHeP13/wOS6NFEIKBpwd58n97lkOfOMK7f3CO0A1YOF/g0p+d5+TPnebARxe5+vnLHP3JE6T2Zfjev32V0rWoLdVYqHPud9/ESJs8m49hpEze+A+v49ejm4iNgo1j3TH2f+gA5/6/t1i8vAChxEiZ1AtNobOAgWeHOfqTJ7n8F+cZ/cJlAjck3hPn8X/yNA//4iMsXJjHq7oIBWJdMb7/715FT+g896vvp/vhXl7916+Q6Evw9D97D7mj+U2J1RookRnL3grvBWHg0qgWaFQ3P/cFCqa2etNQd4u7VhXxQ5eGt3o+WXoSIZQty48MLGIkuS7Ps8gsOdZWgD2cyO9pj7ysKo1ZbHcJXY2hazGSsV7KjRl2qrNSFJ1MYrhp4yCp1Gc61LztLtrW34QQPc1KFUKIGPBR4ALwEvDp5sN+Afhc8/d/3fwzzX9/caf6qlVIliYuUl/cuPy/XVjZXuLdI6tVjvoSlanLHd+th75LZfrqigV/VLUaRI+n2vzk9hD5cq1uRkJVUTuI/vhhQOh7BPbqe6PH0yT7D0Yk5U5Bykjjd3vVsandufU8cisLG+buhb6zUlVbnjjbTO9jF2ciPWEbmOmu9U7SioKZ7W2bjedWi7iV7U3pBK4TrbHN11614hjp/Lq/1xPpthOLUkrcysKmnktLzDPOFQJ8Sszzrnyd6/IC5+X3KLJekO/b9c6NUJfXrxltvcCWkTmYo//JQca+fI3CW7M4Sw5OyWb85Rss3Sgx9PwIRrLZsgklo39zhanvjHP8px7i1D86w5EfP8alPz3P1HcmVsmHhMAJCJwAGUpkKFf+HDjBhi04RVMY++o1Jr81TqNQp7HQYGmshFeLzic9rjPyvv04SzajX7hMY76BW3YoXSty/UujZA7lyB/vavJlQb1QpzhapHxjiXqhTmW8zNK1IpWJMn7Dx0y3DrdW4wa9Hz7OiV/5GA//q58gfqAL1dLJPjqCnum8erhTaEacwSPvI57ePDVgFWJlEhAijdBWK0qbIcoSXG07Ksr2iI+CGvnjbVKNEsu2RHcx6PpW2F6Z6eI7SBmgCJWB3Bksfef7ZVfyILlEtJe7foOp4rld+2y2gk4qVgPAZ5o6KwX4UynlF4QQ54E/FkL8K+BN4Heaj/8d4PeFEFeBReBnd2uxMghYGj+/6yLtZN/BFe0WRP40t2ePtV6YxK0sRsL2ZiVAsxIYidyWp6o6Qrh2SkUg1m+gP6TwGxXqCxNYub6od6/pDD7+CVQjTnHsbHPz3N3zJwy8TSuTYeDdYpgp8RrVDV2cZRisIWBCiQwEN1qpX6/g1kpY2dY6JC2WQjPja6YbNTMeEa52wvXCeEu36ZZotitDt3VrTdVMrFw/1cmra/5eT2TbEqvoGHMtCaZEoqCgoFKhSJnNv9PLmYjpkeOtj3sL/n/23jxKrvM87/zdtfat9xWNxg6CBECAAFdJJEWJoqxI8qIoTuSx4xPPjCf25EzizMnxcWbG2Sb2HE+cSSZnbE1iO7HjkWRrs3aKu0RSBElsxNoAGui9q2vfbt31mz9uobobvQPdQNPs5xyguqtu3fvV7Vv3e7/3fd7nkbUAoZYe8tLJFe9Jka4oejzAto8O0n6oc94lmNiexK7ZaGGNes6/jpyazfk/PUt8W4L7f/EQY6+OMPT1i3jWndlzCE+QvbCwoeEm1JBGfCBOuD3Mo7/14XlZr0AqiKIrBFtCDZK8wDUdPMttBHUOdtVqBnnCE0jKMobhmsLAFx6m9eFBjPEC0V3tqNEAdkCl9zOHyJ24weR3zt7R510tVC1MW+8hSpnVNDqJeRkqX4Nq7VmlxSBJ8rwuNu82lf0dLFwcYiQXLCQkJOKNbK7JvdEQE8JjMn+GRLiHzuR+UpFt7Op+imtTr1Kz1r6gkyWVlugAe3o+hqoEcT2b8dzJe6K6DqvrCjwDPLjI89eA44s8Xwc+ty6juwWOVduAcpfU6OqbPRWuZaBHU2tSNlcCoXmRsSTJaOH47Y9K0ZAV1Q+YJLkhs+A/aqHYmrqXPkjwXJvctVPEunc1uuxAiyTofeg5UoOHKIyc87sDy7nbDxxugfDcpXWQhODmHVcIPzO12CQsxEKb0aXga0XlCSSWJ7ArehA1HJsXWCmBcLNBYykIz6OWGbujxgKzMINj1pYNrCRV83mTc0qekqz4/KoVMoye42Aso7ElIdFCFx1SHxoaV8QZTOq00EGJ/MIJRXjUZsYaQfAqFymSRKitFzUQXlGtXFIkJFmiNlOlMj6/fFkeL2EWTezqLddQYxg3g5v1aPjzr8Fl7muSL9xpGw7lsfmaWOXxEpn30hSvz9Iwbi3draU4EeqKkzzUz9U/eJXy5TQP/IvPAODULMxMhfC21Kr3dacQwsV1zVVlNwTePFX0sJ5CkbXbFvOcC1XWCeuzn7tul28r42JikCNNt7QdCQlXuEjIREiSopM+aSc50vdMeR18mYWhyZdQZI3W2E66kgcIB1qYKpwnVx6eFVYV3jzleiRfvV6WVFQlQDTUSVtsF+2J3QTUGI5rMZE7zcjMW3jCWdHx4ybEnP/vFJtaef1WuFb9jq1jboWsamih6LybVrxvH5GOgTXtR5LVW7SQfK7Vqt+vaOjhOMFUF8F4G3q0BTUURdFDDc0n1Sc2N46jhjamzPjXAZWpa6TPvUbnwaea6t2SqhPpGCDc2kvb7mOUxi9TmhyiNjOCbVTuTORRiFU3Hwj3zg1BbzZlxHp3LzvbKnpwwXWiR5Oowciy+3fNGmZx6azGamBVi373Yrx1yW1uEuwVPdDUkZIUFT2aWjGj5jnmsl6DSdoYlPZhYRIjhYKGhEmb1IMmAkwwvOA9vn1PZdULopsEfC2WWjGwMvN1nJrN1IlJrnzj4kIukQDXnr021LDG3p+9j3B7hKGvXaT/yQEGP7GLK9+8hGcvca2uQ+DlWS7GTA3heJz7z6cxi4uVrT3U4J1PHUokAEJQvjSNZznN9YZwPYTjIet3b3qyzSqlzDCJ9l3Ua/kFi665neee8KjUM4i4z52MBFqJBTvIVhZeU2tFLNRFJNDoTBeCSj19W4GVh8eYGEKVVPrYhSzJqOjskQ4hgDI5boiLd93qZS4UOQAI0sVLxEJdBPU4iXAfsVAXVnsVw8xTt0vYjjGrXC9JfkAlBwhoUUKBFAE1iqoE/KwhNMybi3Ql70OR9YYZ9spfDtMuMZ49tS7n5H0VWAnHXptlzCogqzqyFmTuiVe0AIq2PDdgNVjNyleSZMJtfbTsPEKsZxdaOIGiBbdKe3cAz7GYufgGdq1E58GnCLV0A1IjwPLFMQOJdlp2PohRmKZw/QyFG+ewFiGVrwZiDSa+60U3rDdkRqRlxO9kVV/QmRdMda2obm8b5UVtYtYCz7EwspNEu3cuu50ea0EJhOcFVlps5Y5Au1LEri2tptwu9VIkx6i4wv3SwwA4ODjYhKXYogtTq5zDLGVRQ7FVi9aqwQixnt0YM2PLblcYzlO4XqDvQ/2Mvnq92QEIPrFdVmXcm2U+CbY9OcC2p7Yz9PWLXPqLCyDBvr91gOL1AtPvTs4bv/AEjuEQ7YmhR/Umef12YFVtpt+d5L4vPED7wS5GXhqezZjJEoFkwA+21uEy9uq+7mCgM44xOnu9abEggY4Y5UvLdGSvMyRJxnMs2vuPEmsZwKzlmc00C6auvY5pFBq/uxRrE7jCRpV89fTe1CFKxtQdSS5oSpi+1CE0JdSwxrEo1MZue6I3qDIkzpBmnBRtaFIAB5uSyJMnvST/6m5AU8Ps632WeKgbVQk2hVZv2vqE9CQhfeX7wGKIBFrZ2fVhX8qhcd9fDYrVcSZyZ9aFk/X+CqyEWG96DJIsIysbUVab4/O3BNRglI77Hqd17yN+ee9WA0/Pw3VMhGPjuY7vT+i5gEQg1rJiZ9cHGZ5jkbt2kkr6Oi07j5AaPOx3wzVI277mWIRY1w6iHdtp2/couaG3yV19F2vNHafirouzGrkJPMdeNisqyTJaJOmLWDYWJKGW7hWDdrMwg2PcmYeYcByMzDhCeM2V5GLQInG0UAyr5Jf4ZVVHj6x8Q60X08t23OoEyYs0DhazNw3R9GtbDJ5tUpm44merV3kzlhSVxMB9ZC+8uex46jmDs//xJMd/4zE+8q+fIXM2jVkyCSSDpHamuPa9K1z5xiWEK2jZ08r9v3SY6ZNTDH3jEnbF4vyfnqVlbxsP/PKDPkl8ZJZo7xg26dNT9D7Wx0P/8BEy76WRNYXi9QKjL11f1ee4CeF4DH/vCqndLRz9B8fp/8gAlfESsqYQ6/czeW/8i9cQy5UTVwljokhlKM2uX/0IuRPX0eJBUkcG6PnUQbREiOybqxN2Xg8oqk4g0kKt7HeCa3NsbW66NMxFwRinbEw3iNIyHfE92G6dq+nXsJy1f3cCapRdnR+hLbarIa8gKNbGKBprk365FQ4WWSbJMrnuc+edQJE0EuE+woH1L/fK8r0Pa+79CO41ZikwTRRHz1OeuHJn2QXhUZ5aOjWshqL0PvRTtOw8MmvNIwSOUaGWG6cydQ0jP4VTr/odY66DcB0810ELRhl8+hcIJTtvf3wfEFiVPFOnXyA79Dax7p0kBw8SbR/wVbMbE74ky4SSnfQcfY7EwAGmTv2Q4tilOysPbjDsagmrWlixrKfHUs3ASlY19KYEw9KoTt/gzu/CgnohjWsay45RVjQCqY6mArsWjiGv1BHouZj59KLdlTdRE2ViUpK8mOGmnUeEBFESpMXS2aXSyEXa7nt8RZ/Cm5AkiXBHP5HuHZRunFtm0DB1YoLXfutFBj62g9b9bbRGdMyyxfTJKdInpxCeQAkoDDyzg9JIkff++BRm3s8q1NJVTv/B29z/S4fp/9A2Ln75XLMkKFzB8PeugoD+Dw+w7elBrJJJbXo+f6aer5O/ksOpL8+dM7IGJ37vDfqf3E7v4/10HevBcwSViRJjr43g1GxkVaY6WcEs+dQMz/Eoj5aopf2OXM9yKV0vUM8vnRXxLIfhP3md7k8+QNtjO5Ek6HhyD6WLU1z9g1epjd5Z1nQtsOolrp3+2qq3d9w6I9m3iYU6UWUdRdbob3mQWLCDsdxJCrUxTKe6LO9KkXUCapRUZBt9LYdJhLqbQZXtGgxnfnJPxC23cOf4wAdWwnN9tfM5MPJTpM//+M5Tgku9X5Jo3XWU1ODBOX6HLqXxy6TPvUZ1ZqRpu7Lo2+XNYfz6foJdK5K7epLCyDnCrb3E+/aR6N1LINHuSxs09K8i7QP0PfwZPPervuTGJoVr1rCKWcKtvctup0eTSIoCjp8h1cLLl7mE61LLLF/WWi2scg7HKC8bWEmyTHDOAkGLJBc1vJ0L17aor6CxNc0ouzjIHukQISIMSHvRCVKnRpal9enq+WlqM6PE+vasuoQgKRqt+45TGR9akQOaH8pRvF5AVmUkWUJ4As/xmkGSa7qc/U8nARYEQNOnpsn+kxcQrreAZ2VXLIa+fpFr3xma3e8t24y8NMz4j0ZWDKwArLLFtW8Pcf35a8iy350qXM8vVzYI8Gf/6FQzs2eWTN7+/Z80xUmr6Sqv//NXlyfKA3a+xuiXTjD5rTOo8SCe5WAXDDxz8/sdzpSvMJl/j57UAyiyhiTJJMN9xIKdGHaRaj2DYRf8AMu1EPjZW998OUpYTxIJtBHS48iS1rzeHM/kRvYEherICiNYHglasTAxGgR1GZlWukhIbZjCYJrRe1YOdD2LdPECmrq6BczdgGEW1k2a4QMfWHmuhWsZ80Qc1WDEP8EbFLxooRipwcPI6iyPqzozwthbf7W80nsTUkMAbQtrg2+tU5m6RnVmhOzlEyT699Gy84gfoDS6LgPxNroOPkktM7qywOs9gmubmKXMyqW2cKJhZmyghqIrNj3YtRJ25U5d5n04RgWznPd5XUtBkn0h00ZnoB5JrFji9qz6iuKlVUpcFWfpkPpwG9roWTHFNKPLtpi7Zo3SyAUiXYMoy1jyzPsIkkS0Zyexvj0Ub5xbsSzs2QsDo7lY0kTZE8vyp4QrljVgXum4C/bnCdy6s6SyvGvOeUWAOzdgE8t8jsWOY9gNArtYlzLj3YDrWVybeR1F1uhM7EOWfEN5VdGJKe1EA22AwBNuo/rhd7XNWrjM5/8IIXDcOqO5k4xm38ETt88n1ggwKN1HmTzXxHkEHi10sls67NvaSBIhIgyJM/eEwG67BlenXuHeufktBrFu52IrsHJsrKa2kf9HDsbbkRV13TsQb0KPtfgSD40vlfA8Mpd+smrNK1nRmnpZW7g9CNfBLM2QPp+hOHqBniPPkho87Cu3SxKRzkHCbX2UJ4bu9VAXhxDUCzN4tj1Pg+1WyHoQNRjBqZXQYqll9aGEEJjl3LKk8DUN0fMwMuMktu1fcr+nqrUAACAASURBVBtJktAiCZRACLdeRY3EVxQvtWvFVYl5VihSEaWGPxq4rG6iL41coHXvMUJty2cD50INhOk4+BGM7MQdE/8/SFBjAVof3kHywX70lghe3aF8aYqZH13BGM9vKl7QYqjbRS5PvUTdLtPXchh9TgbGv79LKKsUJzbsAtczP2Ey/x6Ot3aPzrnQCRAiyowYR+Aho9Ar7cTE4Jo4R5wWeqVBxrhKjfX5vq8Vf53LnFutZ0AtMzLPS06PthCIr05R+XYQiLU2S4Dg6xLVsmOs9i4SiLduEdfXC0JgljKMn/g2texYk1enqDrhtv71EQ/aINTzUysab8uq5jsANDwsly1vNYyXl+MurQ2CWnpkxfS6Fo6jBiN+F2M4vrJ4aW5qad2wm/skgIbfzu3irDqoAjDLWfJXT63sL3kLIp3b6Tj8FEpg85Q3NjPkgMqOv/chtv2dh1ECKtXhDHaxRtuHd7P3Nz5OdOcqjLg3AUynzNX0q5y68RdMFs6vSXvKEy41K89I9m1OXv8Ko9l37zioglnldQOfSB8lQYwkE2KYPGky+KT4IFvX6kbgA5+xAqhMX8eulVASfjClheMk+vdTL0ytu7wDsIBD4jn2ihNFE5LsK8WvYKK7hbXBqhaoZcYIt/VxM3OpBSMs5rW3WWBXCti1MtpShs0wq3kmScuX5ADPdail74zXcSvMUgbHqC5r76SFYmihKK5ZW/azgJ9Vq2cnVxR33S7txRIWI1xee3rf88hfPUVi4ACRru2rfpukKLTseQi3XmX61EvrGKD+9USoN0n8QA9X/t2L5E+NNvlZgfYoO/+HJ2n/0C4qVzbAuWID4AmXfG2UgjFONNBOPNRFItRNONCKrkaapb+bpUHTrlCppykak5SMSWpWgfW8z4hGV5aMf9x2erCxmirsXqPAK2/lVjYEW4EVfudYeeIygZjvrSbJMi07DlOeHKKavrHux7uVtyOp2qrFREPJThL9920pr683JKnBRZrNlmxUKXi94NkmZjFNuL1vyW0kRUMNRX0Zg9jyrc2eXcfIrcZ4fPWwa2XsSqHRhblEJkqW0WMtWJUC2gocMK/BLVuJxxQgjIlx25wJq5wnc/FNgq3da9K0k1WdtgOPI1yX9NlXN3VwJSkqsqL55uF3WS4EQNYUXMOicnWmGVQBmJkKtZEcSuj9538qhEe5Pk25nmaycA5ZUpBlDVX27y0CD9ez8TwHTzh3xKNaDg4WDjatUjeSkOmQ+kgzjtkgq6toSMjNAGsL64utcJWbmkensCq5ZikomOyk69BHCaV8ccnVwTfPDSTal+WJWNXivBW3GggTTHWueBw1EKHzgScJJjtW3bH0QYMWSfoCj2sUWA3E24m0zQYoQnh+59km7r70HMv3y1tmUpRkGS0URY8kVpRmMEtZHGN9+RauWfMDoWUgSRKBRHtDKX75MTr16qo4TDUqKJI/edwWhEfx+jkq42uTXZEkCSUQpuPQk3Qefhr1FoHWew//HhXp3E7XkY/RdfQZFP3umR3PhZmtYuVqhPtTSKoMsoSkyOitUQKtUT9bJUuz/95XEHjCwfFMLKdCzcpTs3IYVgHLqeJ45oYFVeBb2swwQTs97JOO4OEyJW40FxohooDAYvMG/u9nbGWsGqjOjDBz8U16jjyLpGpIskyibz9aKEb6/I8pTwxhG6WFKztJRg2E0SIJwi09JPr3o4ZiDL/8Z9hLCE1alTxmMdPoRJOQFZX2/Y9jZCeoFxdJfUsSwUQ7XQc/SmrH4Xn8rC3MR2r7QVLbD1KaHKIyeY16cbpBxl58cpRkhVCqi64HP9ZsKBBCYFUKVDOjd3fwa4TwPMziDMKxkZboYJMkCTUUIxBvW9a3TwiBWUivaM2y9kEKjOwEicEHll1sBJLtqOHYilILTq2MVVmZuD4tRhiQ9tJBL2XyeHP+/i42NitnI/2S3ouE2/tQV8H9ugk/uArRcfgpAsl2Zs68Sm1mZE3eo+sNSVbQ461EOgZIbD9ApHMANRSjPDF0z1wePNPGylTY+atPUnxvHCtXRQlpJO7rRQlr1KdL9P3sEQCcUp30K5fwViEVsQW/FDgqhqhSIkCIPDNU8T1DbwrkTnLjnnoF/nXG1gzdgHAdMpfeQAvHaN//eNMAOdzWz8ATn8MsZzFLGexaCc82kRQVRQ81ibdayH+UJAmzkl+2Bd6ulcgPn/aVwBslwGjndnY8/d+Qv37G17Eya0iqRiDWSrRzkFj3TvSo7yFVTd/AMavE+/avPnMlSYRb+wjEfG82WQui6EEULehb+OhBQq09czaXSW0/SCDW4ns02ibuzX9WHc+uUy+kMQpTS5YRZFUn2jnoE5NvHm/OMbVwHDUQmbO9Rt/xT2PXCriWiWfX5x3TNWvUMmN+gLsEFC1AuH0bkc7tePfbWJUcViVHvZjBqhbw7DrC85C1AHokQSjVTai1By00O3H618JPVsy0bAaYxQyOWUNfRhpAC8cIJDuW7wh0fVPjOzFeXgq1mVF/v8sEVs1O2RVK3P4YVyaVd0r9JGilVerCxpwXWKUZ47q4sKqxV6evM33qRbqPPbfs+VsMsqKS3HGISNcghaunyV0+QT03Pc93bsMgyajBMHq0hUj3ILHe3YRautEiiU1jlxVoixHd1Y7wPOL3dc97TQhBy/Htzd/N6TKZ169uBVZrgINNmoWadAJBmrFlhXK3cGfYCqzmwLXqTJ78AZ5j0bbvUdSAHyhJikoo2UEwsXiXytrLcoLslbcJtfSQHDyILCtIkkww1UV3qtM39W3qakmznWlCUJm+xthbf0Uw2UmsexfSKrlZsqLRdeijJLcdaAy6Ofol3xOItxGIt80b99yHmUtvMv7WN5fsntKjKQae+BxaOLG6Y0oysXnecvOP51oGN17/SwrDp5ccc3NXkoSi6YRSXQSTncT7GjsS8zZqbgv+zdy1DLJX3mHmwo/vCe9krfBFOCvo0aVtYPR4KxHXYbm/tWvVMbKTGzBCf4x2rbRsYKKFokR7dsIygZUQwrfJWUXmZ0aMU2DxwHg5HatFDkr24lsE4m207n9kzd24kiShheO0P/AEqZ2HKI1dpnTjHLXMOI5RWRcOlqSoyA1/Uy2aJNzWR6itp+GJ2Qiob7nWNwOqwxlO/cOvrGpbIcQ8HtYWtrCZsRVY3QLXqjN1+kWM/BRte44TbutH0YNIkrxk5/1N8TfPtrCqBYqj51cUlrRrJSZOfh/PsUhufwBFD80JpOZPgUIIXLtOafQik6ee9/WuhMA2KgRiLav+bDfVxVe77SLPzn9YzU260Qxwe8ecfzwkeUW2m5Gfwiym0RteilJD9LO5o0V2IIRo8JWmyFx+i/z1M5tWGPRWuLa1IoE9EG9Dj7Us+/dyjArWBmXonHoVs5j1uYdLjEGLJImH48teK8J1GlpvK0+wZdbq97g0PNskffpl1FCU5OBBX8l+Dbj5vdYiCVp2HyW5/X6sSt4v/efTmIU0VrWIU6/43cHC87u6BH5TBY1HWUHWdBTN1yZTI3G0cBwtkkCPptBjLb4lkKojKcqyWfPNAuGutTwqNc6F3NCckxuLqNU3GNyErOooeqgpBu170TYM1d8Hi6p1Q0O776ZAsv8or7qhas6OmtUP/1x6swb13iLecX+NsakDK6uSJ3P5rebvZjl3251augY379lCgGUv/d3xHIv8tdNUpq4R7Rwk3ruHYKoLLZxA0YK+95pwEY6NY9awjTJmKUt15gbV6etY1fySWRxF8cdh22AWZxg78S1KE0Mktz9AKNnZ1PNBCFzbxDbK1AvTlMYvUxq70OTA1EsZskMn0KMpPNvEqiw/kQjPpTQxhGPWbuv8LYbK9PVlsweuZZC/empZbs9a4DkW5grE5eLYBerFNNHOQSLt/QRibajhGIoe9CccWfFvnq6D65jYtTJWOUd58grliSHMSn5FwroQglpmtHltulYdu7Z4edIs58gOvY2kqD4hvrB4151rGuSvnWoSzKsrnNvmWDwXIztJcqdYMmiRZHlFEne9MI1rboy9hXtTKV3sXVIXbDVjdIzyssKgCuqSBstz4eGtuRvKquSZeucHyHqQeP/e2w5aJFlGCYQIBUIEW7pBeH7Z27EQro1rm3i2jXBtP2sty0iy4lMPtACSojZ/lxUNWVVnJ8RNj5sLG6lxGTRXaH6ApGrIjQ5p/7Np859rPCpaAFkP+Fk61X/UYysbd88biaKQ2vUgodbuxvk3cW0LzzZn/zm2/8+1ETd/bvzuOY3nXIe5JuzNgHju40bi5mKcWxamNwPxxvmTlLnnUJs9v5reOId6M+sp6/5jYI1etIqm03HoKZI7DzepI/65tPyfHQthW83zt9S5Fa4z79zNNo/cfbP724V0R0bD6zUISdrQQYTDEv/ynyY5csiPwLM5j9/8Z3kuXl5lvV6SUYMRFD3U5F4JIRCui+dYuHYd16qvOCErCvzC5yMcPxrgf/lXBTLZ2e1lVUMNRv2bp6wiaEz+Vh3HrK6K+yLJCsFwK3rDmd2xDYxKBs/dGNkATY8QinVSr2aw6kvznm6FqoUIRtowKjO4zgZ6VUlyk9clK5o/KTXV7n2PSM+q41jGhnCL7hbiA/ex49m/e9sSHEIIJk98l+mTL25YF2Trvofpfeyzy6rELwchBNXp61z99h8uWj4LEGK/9BA6K/GgBFNihBFuzwcykGin74mfWZOX4GZGaewSN174MxxjY0nMSiBEcsch1FAMRQ+gaEHkJufSn9glWWkEjcrsz7I65zm5mf3baDQzLq6L57mIm//cOT97biMYtvAsE9eu41n1Jie0XpimPHJxQ7QQwadaxAfua8xNwQaPNeCfVy3QXEjOP5+L/D43yN0wCPxTeuv5W+R312nMqWYjQKv759eqU50a9iWQNkHcArwjhHhosRc2dcZqvWCagj/5rxVeelXlC5+Pcuh+jUh4DStO4eEY5TtuRVcUOHo4wCPHAqSS8rzAynNsrDvwaFNUnZ6dH6az/yhIEsLzsM0KV05/lVp5fbWJbiLWMsCuQz/DjQvfZ3rkxKrf19pzkG17P8qNC98jPfruhowN8LMBZg13HbN0mxFWOY9jVFYU11wKnl2nnpvaUGkJIzuBZ9dvO7ACMAvpJTlJDjbj4ipyw75GJ0in1E+ZPCXhf68SUgsBwktyr1Y1huIMo698md7HPk184L4Vuxi34EMLx+k+9gm0cPxeD2VV8EtifuBxuwXV4sgFqhPXNoxWEGzppuf4J9etIrCx8LOUkqIu28SyEqZPvUgtM7bpF8IfiMDKdeHUWZtTZ22OHNZ54MC9uRnaNvz7L5b4y2+qXB9Z3wsjHO+hc9tDVPJjjF15Cc91kRWVei27rsdZD1hGgWpxErO2fjyYDzLsahG7Vl6TJMBcOEZ11T6Vtwu7WsSq5G87+EN41DLjS77s4jDTsOkA2M5+smKSG1xulv3SYpQd0gGiJChx+35+ViXP6GtfpaOUpXXvcZRGN/AWtrCFLcAmC6xkGRJxGcMQ2I4gEpbQNAnLFtSqgsV4jooC0YiEqkq4LtQMD2uZytdqMogBHUIhCVWR8ISf8aoZYsF7Y1H/ZlqpCjQNImEZWQbL8rd33fmfS1Ugl/NIz1jN19YLwVASWdbITZ+nUlh6AtoMKMxcoZwf3dgy4AcIrlnDKufWZBo8F1YlvyRHbL3g2ib1/DSRjoHb8l/0XNfPqq0SESlGUWTncalcXFw8wlLsjqkvjlFm+t0fYhZmaH/gw77A700S8Ba2sIUPNDZVYNXZrvD7/zrFd583KJUFn/tsmN4ehZExly9/rcoPXjCoz6kEtKRk/sZzIX7q4yG6OxXyRY/XXjf58teq3Bhde+QiSbB/j8bPfSbMkcM6bS0Kli24fMXmv/x/Vd44YTKXT/w//4M4siLxn/+8wqeeDfPRJ4NEwhJDVx1+79+VuHDZbn6u3/1nKXbtVFEUGB1z+bXfyDE5fWfRlaxoRBK96IEoibadzd/bGxFgvZqlUhhrGoLKik4o0kognEJWdFynjlGZoV5bSNiWZJVwtJ1gpBVZVnEdE9MoYFRm8BbR4QmEkoTj3SiKhm1VqRYncOz5KfBYy3aCYd9WRQiPcu4GprF41kpWdCLxLvRQwheZrGYwyjOIOWrF4VgneihBKXudQChBKNqBLCvUjTy10vSGccs2I4zcJInB+7kdroSRGUes0XB4rfDsWZX428uqLU9cvxV1DFJSB0WRpYpfwo+RJEELWdanNO5adbKXT1DLTtB5+Gni2/Y3OlE3f3B1s5NZuM6alOW3sIUtrIxNFVhpGuzeqTE4oDIx6fKTd0xOvAvPfSzE7/52ilBQ4stfqyEEJOISv/mPEnz6kyFeeLnOyz+qs61P5W9/LsJjDwf4tX+cY3xi7YHLvj0qH348yDunTH44VqejQ+GnPxXi8EGdX/rVzDzCe3eXyr49Gv29KrGoxE9OmASCEm0tMnVz9maVzbv8zr8t0tOl8Cu/GGXnDg1tHaqRmh6hZ8djBEJJ9EAcWdFo6bqPROugf9yp81RLkwjXQ1EDDOz/BKmOvfhEQoGiBnDsGmNDrzAzfqoZXClaiL5dH6Gt5+CcoExFeC43Lj5PZvzUvHFEk310bjvmE1IVDUXVKWaGuXb269jmLCk23jJAS+d+9FACTY9w5dRXFg2sAqEkA/s/QaxloEn8lCSJ9Oi7TA6/gWP7nKlU5z66Bo6THn2Xlq77GscOIEkyM+OnGL30wgcmK2Zkxv107BrndCE8qjOjG0awnXMkv/PQMla01lkMVim3JlX4SXGd3dIh7pcewcX/zipolMmvrzCi52HMjDLyypdIDj5A6/5HCLf3bWrulfBczGKG0sgF8ldO4prvD2mRLWzh/YJNFVjdhK5L/PbvFDl5xs84fOv7Nf7w37byi387yguv1MlkPT7+dIjPfDLEF/+kwr/5DyVs2y+5vfyjOv/X77bwiz8f5V/93upXuODPS995vs4PX65TKvuBkSTB2XMW/+H3Wjh0v76gk3D3TpXXXq/zv/+fxeZ7VIV5ZUvLgvfO21wasnnmySA7d6zPTdc0ilw59VWQJDp6D7Nt38cYufQ8uSlfVVp4TtOT0HUtCpkrFGau+Fkn1yIU7WD7gU/Su/MJStlrzSAn3jJAR/9R0qPvMD1yAuG5qHqYYKSVSn6+zYskK7R03cfopRcoZIaQJJm23kP07XqSUuYBJq+/yc26y/iVV5i4+hqdA8fZtveZRT+TrGgM3PccsWQ/Ixd/QDl3AySJ1q4D9O76MABjV15uBgKBUJLW7vsZG3qZcmEEVQ3Su/tJOvuPUspeJzd1bl3O9WaHWcrg1Gto4eVNjG+FY1QxN5hfdRM3LXPWGlgJ4Vv3rCUAqFHmvDhBnBQhIojGc2XyOKx/ds6z6uQunaA0coFY/z5a9jxEpL0fWQ9uigyW5zrY1RJGZpTC8HtUJq9iryDRsoUtbOH2sCkDq7PnLIZvzAYwwzccTrxr8TOfDtPZrpDLeTzxaADbge8+b2A37pOeBz952+LykM1THwrwb/6DhGGsLc1dr/vcqLYWmUBQQlV8jpXjsmgnYaHg8ZffrDWDKgDnrhmGi2ZGxnUtBKKhx7LIBCQEucnzzCWXmEaR4swV2vsOo6i3tqn7pQLHruNYNUyjQLU4wQIIQTF7jenRE81gZ2bsJB19RwgnupEkeV75TghvngH1rQjHOkm272by2utkJs427T8mr79JrGWAtp4HmBk71STlC+GRmThDdvKsPwEDM6PvkmjdQSTe9YEJrByj6pPD1xBYCSGwa0Ws8u13o64FtlHBLGYJJhd3MFgKwnGo56dYKzHKxly3st9q4RgV8kPvULpxnmj3Dt+Xr2sQLZLw29/vSpAlEJ6Ha5s4tTLV9AjVqWFq6RHMYqahsbdV/tvCBxSyjKQqCGvj6A+bMrAqFAW2PfvFt23I5lwiIYlQSELTobVFoVz2KJbnc4MsSzCZ9jh0QCMWXVtgJUmwa4fK3/zpCAf2a6QSMsGARCwmEwpKi3JujbpgeuauRVJ3BFnRCEVaCUbb0PQIsqwRjnU2lXZvopIfI5++THvfEcKxLvLpSxQz16jXcgt8zoQQVArj80pJjm3guhaKovulqTXcw4PhFhRFp1oan3csz7WoliaJtQwQCKeagZXrmFRLU82SJYBtVRHC25hyjCwjB4ILS26uh2ea3KsJyzVr1PPTBGKtK5cDJRlZ1/EsCyM7uaQchaSqSKoGSD4Xx76FsybLyKrWUAtf+XMLx6GWvkGkc5u/f0VF1nV//56Lt4RAqWvWqBczyHoAz7pzC5gNhxC4Zo3i9fcojw+hx1JEOgYIdw4QbutDj6UaumrKHYp73qIN1MhK1Qtp6vkp6rkpjOxEw9/U4l4GU6JxTm5Xa+39CM+q+4KhGwThOThmbd69by2QVAU5qIEQeIaFAJRwAEkC17ARdy9DsGp49p0HQ1p3B1p3B7V3zmzYV2JTBlYNMeH5zykSQvhZKc8D1xEoqoR8y01JkkBX/VLcWjvverp9kvlAv8p//UqVU2ctsjmPnTtU/v3vLm4d4zXGtNmhBaL0736aRPtOHMvAsWt4noseii9Qq7atCtfPf5dU+xW/rLf7KboHH2Vm/DRTw282OU4+xMIM2c1Jds3zhYSs6Ajh4S6S1XIdEwlfs2v2UB6eM3+ybZJxNyA7oHd00fHTfxPpFpKcNTXJzLe+hldbPQ9oPSE8l8m3vkP69Esrbhvo6Kb18WfIn3iVyrWLi26jxhKkjj1BqG8QZJna8BCZV783L4AK9Q6QeugJZl76DnZhZVkP4TnMnPsxheEzAER27CN55FHUaBy7mGPy23++pD5NsH+Ajgc+zfTz31gY4G1ieLZJPecHObmhd1FDUfRYilCqi0CqAz3aghaO+uLDDVFHWVVBVhrfSz/7JIQ3R6HaxLVMXLOGbVSwKwXMYgazmMaulX3tNuveBfmLwSrnuPb9P940BtB3A15D4HKjUJ2+wbXv/r+3dZ+TVIWOTz5IpK8XO1dh+vkTCNel41NHCe/sZPrrJyidur7avXG3rjXHmC+WLeka+rZe5EgIJ5vHnpgGT6C0JJECvoOJ1tWGky1gT86gdbUTefgQSmsKEDiFEtb1MXBc1K52tM42vLqJNTKBMOqonW0gSciRMEo0jD2RxplZ+V63KQOrgW0qkbBEpeL/saJRicHtKumMS7HkYdtw6YrDYw8H2D4wXxMqmZDZtUNjdMyhUFxbxPPgQZ1Hjwf4rX9e4D/9aaUZMLW2yOj6vedJ3Ak6tx2jve8wY0MvMzN+CtuqITyHbfuepWvg+ILtHavKzPgpslPnCMc66ex/iN4dTyAhMXr5hQ0apcCxa0iyjKotFL3T9IhfvrJqc99yV6cPWdPQOzuRbxG6FLa1pIdcsKsPNZ6gcnljy5J2tbiqzjkPl+rYVWrpEeza4tsnDh0nsmMfuTdewi7m8BxnQVZKuA6uUV2QxVwObr2K2yChW5U85eHzJA4eIzy4m3p+CuEssi9JQm1rwzVq6yZiGk0oxFIqk9c3buJr79Vp69GplV3GrtZxbRu7kseu5KlOXgN8jqKsB1FU3bdvaVjW6CGVzoEAU8N1bNP1nR4aWSnfBsTBtU3EMmX1zQThOneNy/dBgdeQMLkdBHpT6DvCXP/Dr2EXaziFGsLzqH1xjIFf/wSWkVuVvIkcUIk/OEj57Ahu9S5nk1WV+E89hZKM42YLRB49QvWt0xjvvkdgzyDhYwexRycRto0UDODM5JCjYZRkAimgI0fCyHUTSZLQd28n9vEPYY9NIUcjhI8coPD15wk/eIDg/XuwhkcRQOTDx8n/2Tdwc8vzEzdlYHVgn8bnPhvhq9+s4TiCZ54K8cQjAZ5/qc74pJ+G+qvv1fjUsyF+/b+Nkcm6TE65xOMyv/x3onR3K/wf/644L2MlSX4mTFP9kp6uS6gqzL2PW5bPr+rqVGhJydiOYFuvyq/8YpQ7XWg1j6lIyI2fFWXtWbXbRTjWietY5NOXm/Yzqh4hmuhesIqUFa2xSnbxXJtKYQzbqhFv3U4k0bOh46yVpjCNIqmOvRRnrjQlG/RggnjrIPVqhnp184meLglZJrr3fhCCytD5TWHFYOVmmP7+V5fdJtDRTX1yhPLQe4glhOHqE6PUJ0YXfW018CwTK5fBqaygoSUElcvvUbn83m0f61bsfjDKsY8l+eI/vbFhgvPRlMqRp5P07gzyB795nWJmYdAoPNcPNpmf6YxvD/DpL/TzH//XcYozdx48xbt307rjKDNDb1JJX7/j/W3hfQpZQm+JEtndjayrSJqCrGsNCx9w6xbi1klJltBaoqiRIMLzsDJlPMNCiQaJ7Oqk7eMH8UwbO1fFnCrgmXcn2Nc6W9G395P7o6/gFkoE9gwS+9gT1M9e9JNorkv5hdfxyrOd6eala2hdbSixKJVX3/Lvx4pM5Ilj1E6cofbWaeRImNQXPou+3Te2d6YzFL/9EsKyaP2Vn0ftbHt/BlYXLtl87Kkgn3jGz1oMDqhcuGzzxT8pU6+L5ja/8/tF/qe/H+cP/20r6bRLIiGTiMt88Y8rPP/SLF/jI08E+PRzYcIhiaMP+nYy//h/jDMx6ZLJufzRn1a4Mepy+qzFC68YfOHzEY4c0jHqgmRC5twFixujt6+U/nOfCfPYwwEiEYljDwbobFf47d9Mks15pGdc/vCPy6RnNraeWC1NkOrcS+fAcfLTF5EVlVTnPgLh1AKz39bu+4mltlErTWGZZWRZJd46iKpHKN2mBY2saGiBGIqqEwy3IEkKwUgb4VgXrmthmxU816JeyzN9/Sf07Hgcsf8TFDPX/K7A7gPowTgjF3+AbW2sr9m6QJII9W0nPLCL2P5DeHWjmeUyJkYon5+VrFCjcSI79qK3deKZdarDl6hPTUCDtyYHw8QP05/bdgAAIABJREFUHMYYHUYJRQhv3w2SRO3aJWqj10AIlHCE+IEHqV67jJZIEdq2A4DK5XPUJ0f9bUJhEoceRolEQXgUTv4EOz9r7yIpKrH9h9BbOwh29eFZJu1PfhI8j9K5k/5+ADWRInH/UeRQGM+sU3j3DdzqQrsnNZbwP1drBwiBXcxRGTqHU169GGmwd4Dorv3Imo5dKlB458cLb/ybFMPnasgyfPpXuu71UNAjSeI9eyiMXbjXQ9nCPYSkyoT3dBM7uA2tNUry0b2Y4znsXGXJgEhLRuj67DEkVUGJBTGnCkz++esEupIkju0i2NtC4vgu7FyF3MvnsdZhIbCqzxIIgOfhGY0GrnLV90HUfaqImy81X1t+RxJyNIxb8O9LwrbxKjWUmN+97GQLCNsBTyAcB0lZOcuyKQOri5dt/vjPqzx2PEBXp8LXv1Xjpdfq8zoFHQe+/QODoasOH3kiQE+XSr7o8fa7Ju+csqjWGnIJje1tW1C0BS++Mp8PNFe+Z3rG4zf/WYHnngmxrU+lWvN494zFiXcsrl13eO/C/JX78y8ZXLhsU6msHBSZpsCyBD948Zbjr1M8JTzXN6tcYoczY6cJRtpItu8i1bEbxzYoZoYZH3qFnp1PzCNAuo5JONZBsm0XkqwghIdtVZm68RNm5gRWQng4dn2BYKho7GNu91802c/g/Z9CllWfIyVJdA08THvfg3iOxcjlH5KfvggIpm68hRAerd0HSLTtBASmUWTkwvfJTs1mfYTn4Dj1BeRNITxcp75s9+GGQ5KQVQ2EhyTLeLbtl7JgHkdIjSVoe/I51Ggca2YKva2TyM59ZH/8Q6pXff6TEggQv+8wwa4+JE3DMwzkYBA1PmsPIwdCxO8/it7ehawF/EAuGEKNzHYJCtfFLhdQY3Fi+w5TvXppXmB1c9yebfpkdcfBM4xmCaq5H8fGLhUIJ1uI7TlA+fypBYGV3tZJ+5PPoUSimDPTSIAaT1IbHYY1BFaeWccpl4js2kewd4DCyTfXNc0rAfFWlT1HogydrNDSpaNoEvEWlWhCZeSSwfXzNZCgayDAjvsjyIrElVMVilmHnQcjDJ2s0N4foKNX58yPSvTtDlGvuUwO+4LCt+YogxGZ3YejtHZpFDIOF0+Uqdc8gmGZvUejJNs1jKrbpM7ICuw6GKF7R5BgSMZxBBdPVBi/+sHQaNvC+kFYLsUTV3GKNbRUlKmvvIFnOQh76e+UUzaY/qt3wBPonQm6f+4R1GQYYzhNTpHROxOkv/E2Vq6y7H7WG17Vv58qLUncTB6toxXPNBtNRLMiuLdC2C5SQEfSVH+RJgRuJofa3YF1fQw5FkFJRHHzRdSWpJ94WGOlYVMGVkhwecjm1JnlSaqOA+cv2Zy/tPQEKoBXfmTyyo9WV/8dG3f54p8szIj8P/9p4XNf+urqzH3/4hs1/uIbG2sEnE9folqaxDQW58xY9SLXz30bLRBDVlQ818YyioBEuTCKWZttuc9PX6ScH0XVgn4Xj/BlF2yzMk86oZQd5uKJ/9IsLd6E69S5cvIreJ7T7BasFicYOvmVRfnsAtEYy+z7J4ZfJzNxpsG18nlVtlll7hdlZuwU+fTQApHReiXDxRN/tkD5/a7C86gOX6Y2Okxo207q4zfIvv7Cgi9obP8hAq0dTH//q9Qnx5GDQTqf/WkShx7GGB/Bq/vXjawHUGMJ0j/8BlYu45uZCjFvf0oojBIIMfPyd7CL+cY2szcFzzIpnzuJnZshsnPfgiEL16F09m0AQj3bcColsm++iLilE8etViidfRu3ViHUv2PBfiRFJfHAUdREiqlvfxkzPeUbsOqBJTv/loKVmcbKppFUP5t2O5CDIZSIv/oUnodT9K81ISDeqvHsFzowKi7vvVHi4BNxHng8zntvlJEkOP5sij/67RECYZnP/vfdpEdNHEtw+MO9vPClGR5+NkV6zOTYM0n2H4tx46LBsY+neO/1IpPDC+85sgJP/kwb/XtCjF0x2HcsRue2AC98aYZHPpni6NNJzr9VZu9DUaJJ//Y8eCDMs7/QwcmXi+x/MoZV9zj58to0+rawhZsQtotnu362x7QRzvKrey0ZofXp+1HCOkpQR02EkTUF4Xp4tjO7H+vuGiM7mRz1C1dI/NRTuJUaaipO5UdvL693JAT2+BTh4wdJfPoZrNFJjJPnqPz4HeLPPYne04kUDGBPZbBGJwnsHLitsW3OwGoLa4ZjGysGEq5j4joLb/ZGZWbe70J42GYZ21xY3rl1f7XyIuTJhgXN/G3r1EqTy+5v/j48rHppQdA2F7ZVxbYWduF5no1R2fxEWUlVCfXvwDVqyMEQob7tALiGQWR7L0oo3AyshOdRu3EVK5Nu/L5w0SEch+rwZex8dslt7gbkUIhg9zZqw0OYU75vpQCEsbGLi0WhKCQe+xDxI8cAcMplpr/8p4BLMCzzc7/ezcglg5f/MotV90CSuHHR4Dt/NI3rCP7+7w3S3qez+1CU/LTFV//vSTxX8Lf+US97j0aplhzae3XiLSrToyb9u0OEozLZycUXe6kOjQefSvDGd/JMDtexLcGjn0xx+tUiDzwW5+W/yPDOi0V2PBDmC/+kH1mWGNgfZnLY5EffzOF5sOtQhFJ2dhKTVZ1oxyDJvvsIxtsBgVUtUJy4RGH03KwUigBF02nb/TDJXr+8Wi/OkBs+SWXmBnMXLVooTsvgYWKdvlVWvTxD7vppn581J0MsSTLRzkFSAwcJxtqRZBmrVqQ8dYX8yHu4ljFvnPHuPST770OPpHDMKoWR9yiOX8S1/YBbDUbpO/JJylNXMMs5WnYcIRhvxzFrFMcvUBh5r7nt+w1KIoHakuRmHcXJ5XGLywfIcjiMkojjZLILFjh3C+3PHQZg+msn0FIR+n75qdkXPeETiO+FCK7rUX7hddSOVuRQALdUwc36i+z6e5cxh24sGmRZN8bJ/9k3kKMRvHIFYTvYIxMUvvQtlFQCYds46SzCsqn+pEHZaFSCSt96Ea+8cuf3VmC1hS3cI0iygqxpBLt6fS7TnOyTVcjOq1MLz8Ozlp9QhOduCp0nSVaQNA23fu+tUuRAkPDufWgtrQA+B0NRAJdte8Pk0xYjlwyEN8eCatLCsTwE4FgCRZEIxxWqZRfX8bcr5RwicYXMhMX+4zEcWzB6ucrg/WE8D8r5xVfvelAm3qIyeCBM5zafc/f28wUkCQIhhWIjYKpXPSzDw/MEE1frHPpQgmMfS7LzYJjRywZOQ+dP0YN0H3yGth0PYdUK1HITgEQg3k7UNimMznaiSopC++5HAEFlZgRFC5Ds20+ybz9XXvpjanlfADiY6GDb8c8SiLZSmryMa5uEW3rZ+aG/w/ip75G5+nbzWo1172bw8c9TL6apZn2roFCig9YdRylPX2sGVpKi0XPo47RsP0QtO041O0og2sK2Y58h1z7A+Okf4FoGsqwQae0nnOpBCI96aYZaboJo+za2HfsMgWiKiTM/vKMmEEn3qQjCvLvfleCeXcSf/BByOIyaTFD41ncpvvTqsnyQ2OOPEH30YXJf/irGxUsbOj69PU54Zyd6R4Lovh7cmkn10gR2rkJkbw+JYzsJdCaQ9dmwwS7UEK5H+ycPUx/LUTp5Had45wuoSEzm5/5ukslRmxe/VWZJK1PXxZlcuIj2ylVYKgASAiedhfT8Jii3UGryrJrP5ecHvs70LdSJJbCpAivHham0S6HgbYbmqS1sYZ0gFl3ReY6DW6tQu3GV9At/taBMthmCpNuBcHw+mZ5s8X2m7qHQm97WjpZMLvra8Lkq3/2TNB//QgfFrMOJH/jlcOEtlHWcGK6z58Eord0atino3Rnk0jsVctMWBz8U5+yPS4xfq/PgU0muna1iVF0UBTRNQpYlVF1GVqBS8LlXF98uc/7NMpIs+eUJS1CYsdlxf5iJa3V6dgQJxXz5jukRE88VJNs1zrxW4vLJ2Qkj0buf1sGjFMbOM3HmhziGn2WWbwq7zgnOZVkFSeL661+mXs4iSTKpgYNsO/YZ4j17qeUnkBWNtl3HCSW7uP7GVyhPXkEIgRaM0Hvkk3QdeJJK+jr10gxIMrGOQYTnMvrOtzDyfnu+rKgogRBWdbZEH20f8LsSL7/B9MUf4dkWsqLS/cDTtO44Smn6GoWRs7N/t2gLY+9+m9zwKYTnEoi3sv3Rz5Ho3cfM5TexjeWz6csh8tARJAnKb7x1V6/N6snTGBcuEtjWT9sv/Pyq3uOWKzjZbJOfuV6wZkpknj+LcGc1B5WwjpaKUPjxJYTrobfGqKkK2ZfPYxdryAGNwltXKL83ilPyA2anVGPyS68T2d2NcD2Eu0yQmJDZdSDA2bcMFlNVmYtEi8LTfyPG0Lk6r79QpWK/D8Qi52BTBVbTaZdf+fUsdVPMMzHewhbetxACt1ol0NaJEokiLMvXJLIt8FwqVy7S8vCHCQ/sojp8CYRPaEcIzPQiFkJ3AklGUmRfTV2SkBQVSVV9cuZaJhhZRpJl//2ShKRq8/bj1g2MkaskDh0ntu8gtRtXfC2qSBynXMStNfiKiuLvQ1H88aiar0vmuc2MhKQovlimoiBJMrKm4XnevG2Wg97V7XdB3gLXFdQqLjcu1njlqxme/Nk20iMmjuU1s0EAtunhuoLTrxbpGgjws7/Wg/CgWnI586MSelAmllCZuFYnO2mhKH7GS9UkHnmuhV2HIrT16Dzzt9oYOlnl5MtFXvt6lmPPJtl71B/XlVNV3vhOnrd+kOfpv9lGz44gAihmbISAcFwhGJZ98/k2jYF9Ia6cruI4kOjZi3Ad0pfewKrkmuNerHHD81yK4xcwCn75XgDVzAiOWUOPpgBQAxHi3bup5SYoTQ41mxasWpHc8ElS/fcTaR+gXsqAENTLWWRVp2XgIFnhYZYyuHZ9Qbku0bsX4bmUJi4jXLfRFCMoTV2hfc8jRNsH5gVW9eI0xfGLTfFfs5zDyE8S796DrAaA2wusJFUltH8vTiZ798tXjoNXcXCKJb+EtgpU3z2Fcf4CbmV9hYedfJXCm0OzTwgwbmQwbiyekcm/triQMJ7AGJ7BGJ5Z/PU52H0gwM//dyn+tzMmzgoNX/kZh6/95wJTYzZG9f0VVMEmC6xcl6ZO1WJQVejokIlE5QXfibExl1p1KxjbwuaCcF1K507S8dGfou/zfw9hW5QvniX/1qsAVK+cQ43GSB55lNSxD4Hkc6XKF05jzkyuo+6VRHT3fcT2H0SNJdBiCVqfeIZ48Si14csU33tnVcGVpKjE7z9CePtu9FQraixOx0c/hVMpUbl8jvLFM+B5FM++jRKO0vrYR2l97GmEEHh1g5mXv4tbqyAHQ6QeepxAezeB9k7UeJLuT30ep1qhdPYdjLFh5ECI1PH/n733Do/jSs98f6dCd3VuoJETCeYsKlCiqJw1kjw5ej3OO37uvb723b2+Xvvaj3fX613vPg5rr3299jh7xp7ZGY/SSBpJM9IoDUWKpJgzQYIAiBw6x6o6948CATS7ERpoBsh8H0EEqqurTsXznu+83/vdh7uuEXd9E1ogRNPTn8dMxokd2U/20sW52+py4W5tK2vcevZgkv6uLLYFZw4kGRvIk03ZvPvCGNKenoV97s8GSERNsimbF786SKTJhVBgrD9PJmWjqvDV37zI6EAeqyD5u//USzJqYpmSriMpes9k+OG3nc4qHTeREg69G6P7ZJpAjYZlSSaGClim5Nj7cfrPZ/EGVGKjBXS3QiZhsfNjNfSdc4hKoEbjtodDvPRXQ5w/buLyhshnYrMavc6EtE1yieLpD9ssIKU9VWpGdXvQ3F5Soz0lLvj5dAzLzOP21yAUBWlbRHuP4Qk1ULNiGzUrbiE12sN49yHiA+dmVEQQGMEGdI+fFXd/tjiKpuqouoFu+IrKauVT0WLXcmljm3mH0C/GadzlQovU4mpuwtXWCoB308aptuQv9WNFp8+hq60VadsUBodQfT60xnqEqmEnkxRGRpH5PELXcK9YgZVKURgcmlFxQqA3NaL6fOR6eiqvSacouNpaUQMO8Zamiey7NJUFtxyhabDxFoNwRF1QRY5MWvL815ZvgsYNRazmQm2t4Bf+Nx8PPuSeDJ8Xf/4bvx5j3wfLw4X4Jv4lQZI6f4q+8WE0fxBsm0J8eorEzueZ2PcuidNH0QOOfYKVSTvrTBIdM5lg6LVnsZKzj9LNRIzBl79NITZbQWUnAmbnSnVPZjJRQuBG3n51ynKhaCu2Raavu2wJm5nHZaWSjLz1PWKH96J6/U7kLp2aap8sFEieO0Wm98IV25fkow4RkWae1LmTZHq6rljHphAdZz6oXh9GW0fZyEQmaZNJOuJ+24aRvvJC/5FL08uzKbvE4sCyoP/89LKhnmkyMHN5cfthYrjAxHDx+8q2YLS/uB3egMqmOwO89rVhek5nqGnU6dzsxe1VnA5KiIrItz2fVcXUtsr1fpP1Umfszspn6Dv4PUa79hFsWkO4fQsr7/4csUun6D3wMmZ2+p41cxkmug9jlSnzkokNFdXVs21rujRVFeBqbyP0yINokVq0UBDF7UZvqJ/6PPry90jPIFbBRx5CqArpI8cI7NqJGg4idBd2IsHoP32LfN8lFL+fyBc+Q+bsOSaef2naSFdRCN63C2PdWob+/K+c6FgFEKqK95ateNavRQ2FEEIw8vdfJ3u2a/4vVwEdq3Xuf9LPhm0Giio4eyzLGy8m6L1QmLo9Hvl4gG07PHzn76LctsvDbbu86C7BuRM53ngxwcVzeaR0pv+e/GyQTdsNttzhwRdQ+N2/bpnSKp44mOWvfn9sartbbjf4V/97LYbHuf/2vpXm2b+Pkr9iBkso8K9/JcJQv8n7b6TY9YiP7Xd7cLkVzhzL8uI/xhi/jjV8lw2xeuAhNw8+5Oab/5ThzBmzZHB99sy1TfW8iZtYMKSkMDE2la1X7nMzNoE5CymSljmVXTfrLszClIHnbChExxdESADyI7OUs5CS/Ngw+bEFZF3aFvmxERgrnSZwjqlvzq9Ly5r3mOaCq7GpyOtrOSKbtnjnuTHuerKGu550putO70/SdTiFbTrZf4Gm1bh8IQqZhfuDzQYzl6aQieMORJwC2zNUw25fGEVzk02MFUWdkDbZ2DDZ2DCj5w9Qv24nrbc8TuzSKca7DwGSbGwIb20rsUunSI0t/pouFvmLPYx941todRHqf/onSB8/Sez7b05ljdllEi3cK1egGAbJ/R+S7+tziob7fRRGFyZgXixkoeAI21/7AaFHHsK/c8dV3d9MbNzu5t/+TiNSSk4eymJZcO8Tfu580Mf/959GOHbAGSw0tmrc96SPlg7NIVQnc4RrVZ7+QpDtd3n43V8Zor+nAAKyaZuuUznaOnWE0Nj/Xpr8pNF3f0+haFww0Fvg9WfjtHXqfPLLjni9XN1uIWD1RjdrNrvZfJtBQ4vGQI9zr65a7y79wjXGsiFWbW0qx4+b/OPX08xSYeMmbuImbmIK7tYOFLdxvZuxJNgWHHgzyvE9CRTV+TuXsadG/NFLJwm2rKNhw70MHPnBlKhb0XQUzUUuOVFRfUUzlybae5KGDbsIt20mdukUUtroniCR1XeQS46RGu2ZXFvgqWnCzKUdTZVtIxAU0nEn2jQjUhjtO0lk1e3Urb3TKQKfdXR2QtXQPUHyqYkia4ZK4fZruLwzemAJqYk89uR5kqaJlUgiDMNJTsjnseKJOc1mhUsndeAgyX0LmyavKqTTxtnKSS0Wqi5w+zVUVSGftcglpwMSLkPwxa/UYJqSP/qtYbrPOPteu9nNr/1+I898McSFM3lSCedcBEIq2azkj//DCCMDJpou+OzPhPnkT4bYdqeH/p4CiajNK9+KY3gVOte5EELwwteiJCe3ceWtOTZs8cOXkzS1aTz0dID5sHG7wcvfjPPXfzBGbMJCEeA2FKIT17c6w7IhVtGoxDTltdUbCuEIfhXBdGhcOi+NRbixXjMI4QiMRZl2X277tYCiOKLkcu2wZdUK6i4Ky/Xa3ugouvdg5rl1/rs251cxPLhbWmCWwtjLCbYF6UT5jiJ+6TSjNfuoW30HnnAz2fgQoODyBsnGR+jZ90KxVmkeSNtk7PwBPDVNtN76MUJtm7DNPJ5wA7o3TP+hV8klncirUBQaN96HJ9RINj6CVciiujz46jpIjfbMIGCOSH7o5LvUr9uJL9LmiN+R6N4QQtHo2fss6fG5o7KzQsDOL3Vw+yfbphYVchbf+tXDDJ1bfPkrO50me/7Cdc1qrSZ0Q+HWj7ey7akWvGGdwTMJ3v7LLuccSWjp0Nm03cOr34nTd2E6UnnhTJ6erjybbzMI1ahTxCqbttn9/dTUurms5PAHGZ75UoiGlmlqYVlgmdJ55UuJaYI1zwRTuYoF5ZCI2rz2bJyhS9MbTKeuf8mrZUOsdr+X4+5dOp/8tIf3f5Qnm5VF7+ZYzJ49kiUErvpGtND0tICUkvxAP1aq9METmoaroQl3Syuuphb02loUw+OU+8jlsOIxckMD5Pp6yA8NlQ0jLxVaMISrsbjGWGFinMLo7NkXqs+Pq6ERd2s7roZG1ICjJUAIZ4SWTFCYGCc/MkRhZITC+Ch2trpme0LV0GpqcTU04G5pQ6+JoPr9U/WbZD6HGY9TGBslN9BPfmgAMxa9ZkTGubaT56ipGT1ci+KZvrZmLEp+eMi5toMDJRYIUi7wiZ+1AQJXfQNaqNgCQEoojAw552KJ0GpqcdXVFy2TgBmdoDBSfeNUoeto4Rr02giuxmb0mlrn3jMMlMlsQVkoYOdzWMkkZnScwvg4hYlxzFjUeQar0HkphoHqC6AGAhit7Rit7SVCZ6FpGCtXodfUVrz9QjRKYaSMIe51hFXIMnD0DRJDXYRaNuAORJC2RWq0l9ilk1NTeZdNO6+cLrStAsmh82Si01O/+dQEPXufo2bFVgKNq9ENP8nRHqI9r5Aa6ZlRUspm9OwHhDu24A7Uohk+rEKOkTN7mOg5Qj45PbUtbYuhk++RHO0h3LoBI9gAQpAZ7yc+dH6SaIFtmSRHusnGR4tKVUkpycRG0AbPY5mlL3pfjYu6lb7pY0ibaO75a7rNBWlaS/C6ug6GmfOgbUuYR39xLf6IM1XWuMaPAJ7998coZCxq61QCYYXHPhlgx/3eovdcc4eOaUq8/ulzms1IBi8V6wSzaRtpS7RrxCwSMYvhSzeeDGjZEKtbb3exabPOQw8bDA5YDrGa8flv/Uac/fvKi9eFphG69wFCd9w1tcw2C4y+9DzxD96fXk/X8axcRXDnPRjtK1B9/tkzUKTEzmXJDQyQOPIhqeNH5xQXVwrv+o00fOrzM3Ynie/bw8jz3y5ZVwvX4N92K/7N23A1NDpkag5IKbHTKfKjI6ROnSD2/nvIJXomKV4v3jXr8G+9FXdrG1owhFBmf7FdjgwVxsfInD9L4tCH5Pr7rpq7sHC58axaTfD2uzBWrJzz2kopkbkcuYFLxD/cR/rUiSkCLs1JQfc853jWdug6oV33E7rz7uJ92jaj332W2N7di9ruFBQF/9ZbqHvyx0q2H9vzI0Zfem5p278MIdBrInjXb8S7YaMzcAkEy2bglcPlLEEzHqMwOkr6zCkyF7oojI/OT7KEQPU7BEqvjeBubHb0VKGws9znR9H1sqJ11een8dNfWMwRE9vzHiMvvVBcYPQGgG3mifefId5/ZtZ1ZvvczCbpfr/0nWLmUoyc2cPImT1z7FmSHOkmOdK9oHZK2yQ5dJ7k0PlZ1zFzKS7u+U65LzN6dg+jZ+dqTyWoDvGRUhZlM4LT3wjjxpuCbtsawhPUp/4WQtC8MYivxkU0k5kyUD/+YZYzx7IlY91sRjI2PE1ibFtSKFzfyL5tOe240bBsiNXxYwX+6A9mD+v29FT2shOKiquhcepvLRSeKn2heJ2Rz5xpvUKgGB6MlZ2429rxb9rK+Buvke29eNWiLyVCXEXBu2Y9tQ8/hrulzfEFWsBcqRAC1efH8PqwEnHiirL4IIyq4lnRSfi+h/B0rkLorgW3AVV1DBwjdfg2bSV57AixPe9VPaqiR+oJ3/cA/i23oHi887ZPCIEwDIyVq3C3tpPeuJnoO2+S7e1BWo67+eXac/8SoQZDBG65leBtd6LX1V8x9bcwCCFQPV4Uw4OroQnfhk2kThxl+NlvzVtPUPX6qHv6E3jXrJuMyCpTJGoxqfjXCm6/hu5WkDZkEoUp/c9NXEPYEiwLxXAjVNUpwrtIyIKJLBTQasMohhtrcspEq63B1dI0z7evPRS19Nm4nFgKMD5iEY/aXLpY4Pmvxcjnr7g/5RK7Nnvy3bq0QOKywLIhVidPmJw8UcWQnxDotXWgqmjBEJHHPoZv01aUySmrhW9GOJGu1WupDwQYfel5Mt3nr8q8vOr1IlQNaZkIVcO7cRN1T/4YWk3t4joUKcn2XFy0w7dwGwRuuZWa+x5Cq60tGbktbCOOBkv1Bwjt2Im7pZWx118h232+KgRVr290OuFVa6aMKBfeNIFwufBt2IwermX0lRcpRMenqqf/i4OiYHSsJHzvg3jXrEXRXUs2Wbx830ohyPb2YBcWINZVVWe60bt8yK3qUnjg51fRvi1MajzPD/70LKPd1TV9vIn5YafTmGPjGGtW4dtxO+bwMELXyfcPYkUrm4a302nyvX14t24h+MB9ZM6eQ3G78W7bguLxFAvPFQXF60W4dPSIU5FACQTQ6+qwc1nsbBaZc9YXuo7iMRAuF2rQj1AVtNoatEgtdj6Pnc7MKbqfDUNdSQpZC1V33tNSSoYvpEhHnVmCSxcLHN2X4d7HfRzdn3GyAk2JogoCIQVVFfT3FhYljbVtiEdtAmGF1g6d7rN5x7PPptRKwRlzT5I+gaoKEHJpEoxrjGVDrMA52StWqnR2auRykg/2Ol4ZbkOQTFR21oUQaMEgek14PgeqAAAgAElEQVQtNQ88gm/LtskyEEz7p1zW08wwfrv8U6LdUBRcjc1EPvZxRl78Drneuc0LK4XTybtRDAMrk8a7bj11T/6YoxWZ0ZbptssrNzC1ncuw0mly/X2LIoHC5SK0cxfhex5E85c6WyMnp2oXeA6FEKBpGB0rqf+xTzP60nNkLnQtiVyp/gCRJ57Gu2Zd2WnJKTG/lMXtAycKM/m3UBRczS3UPfVjjL328rItNbMkCIF3/UYijz6Jq7F5/mle55eSbTj/lJIxKxEn3XVmwfeilLbj9D5Pm6/cl5SLH3YvxVcpUOdi40MNNK4JEB3I4PYtq1fvRwZ2JkP8nfcIPfoQ4ccedmpwZrJMvPhSMbGyrfnJi20Tf/s9FI8H323b8d62HTuTIXv6DKkDBzHWr51aVYtEiHzh06iBAELXEbqO/45b8WzaAKZFcv8B4m++DYBnyyZCjz6E0DRUnw9huAl/7HHsXB47m2X8O8+T75nbqqQceg5OcOilfjY92ojbqzF0NsF7f3+BXHrSXT8n+ac/H+eX/2MDv/wf67lwOk88auEPKrSucPH2Kwm+8dUJzEUQq0JecmB3mkc+HuAXf6ueM8dyCAE9XfkpI1CXW3DbLg91jRr1zRqhWpXOdS5+7Esh4lGLvu4CR/dnrmvO00KxbJ5uw4DPft7Dl3/SSzCoMDRk8XM/HaWlVeXLP+Xlv/7nOOPjlb341GCYmgcfxb91+zSpsm0KY6Nke7rJXeqjMDHuiNMnoyru5hY8natxt7SVRLeEELhbWok8+iTDz38bc2JhnkELheJ2o3g86PUN1D7+FNoMUiVtGyuVdETB46OYsaijm1JUVI8HvTaCFqpBC9egeDwIITAnxsgPDiyiIQqhO++m5r6HUb3eko+lbWNOjJPt6Sbb10NhfMwRyQuB6vWh19djtK1wtFihcAnBcjc2UffMpxh58Z/Jdl8o2f5CIFxuwvc9hG/9xvKkyrLIDw+SudBFtq8HMxYD20Zxu9HCtRht7RgrO9Fr66bcnl3NrUQefxqlzDF/1OFZvY76Zz6FFq4pHx2V0rnusZgjTp8Yx0omnGk9oaC4DbRAAC1cgxoIooXDUyVxALI93eSHFiYMt7NZ4h/sIX3q5KzrqD4/gdvuQPUUXys7lyVx4AOsVOXRotylvkVnsjauCRBsvPF0N/8SkTlxinxvH2ogAIqCzOcxJ4qjVdHvvY5wuea9TwoDg4x989uOkaeuOwkw4xMohpvkvgOYk6ajVjTKxPMvzZKlKrHi0zKX7LkuzLHx8tFg26YwMn/5mHJIRwu8+oen2fONHjS3QmIkR2KkeJB47kSe//xvBtn5oI/NtxvUNWokEzY/eCHBj36QnKrxN9hncnRflmS8+HlIJW2O7MvSf7FYKysl7HsnzR/85hD3P+6ndYVOKmGTnlHaxm0I7nnUT3OH0xdfmLR7uOM+5xk+tj/D8Q8zWJNJxWeP5/AFlMUE7646FkyshBAqsB+4JKV8RgjRCXwTiAAHgC9LKfNCCDfwD8DtwBjwBSll91IbeuddLj77OQ9/+zdpslnJT/20c7JHRixWr1bpWKExPl6Z8Fnz+wneegfgjEatRJzY3t0kDu6fNVMtdewwimHg3bCJ2oceQ69rKCEGns7VhO7axfgPXi1xrl4KFJcbd1MLwbt24WpoQgiBtG3yQ4MkDh0g03WG/OhoeSG6EKheL1ptHUZ7B76NW8hc6MJaRFagd90Gah54pIRUSSmx4jHiBz4gcXA/hfGx2aMDk/qq4I67Cdx2B4rbmD6PQuBqbCLy+NMMP/etyjVXQuDbsIng7XeWFVObiTjRd98iceQgVrx82YT4B052XWD77YTuvBs16Dggu1vbyq7/UYaroZG6pz8+K6myc1myPRdJHDpAtuciZmxi1vteaBpqIIirrh6jYyWe1WvRayMkDn+4YFG4zOdIfLhvznX0ugZ8G7eUECuZyxHbu3vO7NpqQ9UErZtDGP5lM479aENKrHjC8bGaBZW4pduZLHbmigLqSRN7Rn0/WSiQ71uYnYSdSJJPLN4mYi7k0xbDXXNve7DP5Pmvx3j+67OXlHnzuwne/G7p+es9X+B3/q/y5sKFvOTdV1O8+2p5spqI2fzBbyzsXS9t+Kvfr8zR/lqikif9l4GTQHDy7/8G/Hcp5TeFEH8O/BzwPyf/nZBSrhFCfHFyvcWl4czA5i06586aPPvPGTZu0qf663hcoijg8y1S6zFZEsKcGGfstZdJnThWUiPrStjZLMnDBzFjMeqf/iSu5pbiDkdV8W/dTurEMbI93YtrVxkohkHNQ4/hqm9w2lEokDz8IRPvvOmQmLmmRiaLAVupFLm+HhIH9zvHXmGGkxoIUHP/w1MC/+nNSwojw4y9/grpMyfnJ5SWRX5okLHXXiY/MkTtg49OkRdwCKrR1kF4132Mfu+7FRnlqcEgwR13OTYKV7TRTqUYfeVFkkcPz3vs5sQ40Xd/SH54iMiTT+OK1M+5/kcRimEQvu+hKSI/E1JKCqMjRHe/Q+rYEax0at5pNmmamBPjmBPjpLvOorz/Lq5IPfmrYANxo8Dl02jfFiof6buJm7iJjxwWpDYWQrQBTwN/Nfm3AB4G/nlylb8HPjn5+ycm/2by80dEFd4oDme4YjMC/H5nWS63eP2Dnc8T2/MjUifnJ1VTkJJs93km3nmjxGbB0W+F8G+9paoV1IWi4GpqBlV1CvUe3M/Yay87o+9KdFJSYmcy2OkKi3oKBd/GLRhtHSWdhJ1KMf7ma6ROn6goSicLeRIf7iO6+52SbDChqvg2bcPoWFlRM72r1uJuKfUwkqZJbP8eUseOLDw6YpqkT50g+u5b/yJF657O1fg2bJqVVI2+/Dzx/XsdO4oypEooAt2n4wq60bz6pCGrA1UTaKqJHRtA2NPEWTXUqfUvP/KaoaHoCrrfhe53IWZkOCm6gitQuvxGQajRoH5VGR3iTdzETXwksdCI1R8Bvwpc9piPAFEp5eUetA9onfy9FegFkFKaQojY5PpLKrB0/FiBx59w8+THDJJJiaJCY6PCPfe6yGbh4sXFT7TmBi6RPHqo8mk7KUmfPkVq9QmCd9xV3PkoCp4169GCoaqYPgKOGBdHw5Q5f46JH36/rMHp1YLq8xHYfjviCvc3KSWJo4dInTy+qGwVWSiQ+HA/7rYO/FtuKZoSVH0+grftcITsC9i2cLnxrt+IcoWPjJSS/NAA8f17F06eL3/XMkkePYR/yza8a9ZX9N3lDMXw4L/ltrLRSTuXZfyN10ifPT1nlKppRwvtD65A0VTMTIHT3zpBoi+Or8nPmo+vw9foRwL9e/q4+P3zBNqDrH56LZ46L1be4tyLZ4hfjLLhi5uRFvia/WiGRv/uXnp+2I3iUln7qQ2EOsMgYXB/PxffWJwu72qheUMAb7iybOOrhaIiCJex1DT6ihsx2YRyHPgatOV6n4Ny+5eSZZX1VilKjvla33PXGPMSKyHEM8CwlPKAEOLBau1YCPEV4CsLXX//vjwvvZjl//glH4YhqKtT+B9/GiaVkvzJHycZGV6cqFTaNsmjhzETiytgaueyJI8cxLdxM5p/uraREAJXXT2u5pbqEatJWIk40ffervp254OxohN3U0vZ9sT3712SuaeVSpLYvxfvqjWovunRvVAUPKvX4mpoJD/QP+929JoajPYVZaNVqeNHF51QYGcyJA4ewLNq7ZwZcR8luJpb8KxcVTqFJSXJo4dJHT8659tRKIJVH1vD0IcDDHzQj2qoZMbSCFWw8UtbMLMmR//uELZpY+UtNI/G2k9tID2U4sxzp2i4pYlNP7GVg3/6Af7mAEIIjv7tIYIrQqz99AZGjg7TeFsTvkYfx//uMO6wm20/fxvRc+OkZpeHXHUIReAJavgjboINbjY/2oR7Rh07zaXQujlYXNtuFkhbEu3PEB1YfIUEza0QWeGjZUOAlo1Batq8GAHn1Z9LmkQHslw6EePS0RhjvWnMXPXTroQCwQaDupU+GtcGqF/pI9joxu3TEIrAzFnk0haxwSzjvWlGLiSJ9meJDWYoZCtvj5z6nwNFE9S2eWnfFqZtS4hwi4Hhd0TSmUSBiUsZ+k/G6T8RY7Q7hVWB8aWiCuo6ffhqHPJs5m2GzyXIzSitohsq9at8tG0J0bIxSKjZg8ujIm1JOlZgvCdN79Eo/SfjxAay2NbiWIduKDStC6C5F2bUGx/KMtabviqkTnMrNK4J0LE9TPOGIKFGN5pbJZcymejL0Hs0Su+RKOO96aKckEC9e8pFfyH3f6DeTW27d8qnS9qS0YtpkqOVzzDUtnsJNRpTJNAq2Ax3JckmKhuMLyRidQ/wcSHEU4CBo7H6YyAshNAmo1ZtwGVl3iWgHegTQmhACEfEXgQp5VeBrwIIIea9rNks/MPfp9m9O8+WLRqBoMLoiM3RowV6lhCtslJJRwe1BN+pXP8l8oMDaGuKi0YKVcWzag3pUycWve0rIaUkdeaUY0R6DSF0F97Va0tc3aWUZLrPkx9aRHbhFcj2XiTb14tv/cai5ao/gHf1OieDcZ5hjrulDdVfWrzTSiZInz21pGFStqcbMxZdVDmUZQch8K5ZV0RyL8OMx4h/sHveyJ+0Jd0/OM/Kx1bjbw3Sv6ePZF8Cl9+Fr9nPia8fJdE7PaDxNfmo21RPpsFHoCOI7tGdKT6fC2lJBg5cItmfwLYcCw/N0Gi4tQlfo5+NX9qCUAWugAuj1kMqdm3LXNS0elh5ey0Nq33UrfQRavIQbHDjq3WhqMW2D/6Im0/++y0L2q6Zt3nzz87x1l92Vdwm1aXQcUuYu77YQeftNfgj7qKp2JmQUpIYzXHhg3EOPN9H9/4JzPzSCZaqCxrXBLj14y2svaeemjYP+gI6fatgkxjNMXw+ybHXBvnwhUsVGaraBYllObVlG9b4ufNzHWx4qIFwszGr1k3azjk4+6NRfvQP3QydTSzodeH2azz2S2vZ/IhjCpocy/GtXzvMud1jCEXQsT3Mrp9YwaodEbw1+qz7ty3JWG+aE28Mse/bvYz3VijVAMKtHr7we9upbVtY5vLur3fzvT84jVWFa30Ziipo2Rjknp9cydp76vCEyh/zXXYH0f4Mh17q54P/1UN8OIeiCu74TBuP/Z/rADBzFm/++Tne+ursbv0bHqjnqV/dOGVhYuZtXvjt4xx4rnJLip1f7GDXl1dOkbTUeJ5v/Mohzu+tTCg/L7GSUv468OsAkxGrX5FS/ishxLeBz+JkBv4U8MLkV16c/Pv9yc/flEsxgJkBy4LTp0zOnDZRlOrUcrUScUf4vQTY+RzZixfwrllX8pm7uRWhaVXLDpT5HJmus0suQVMpFI8Hd1t7qWbMssh0na2KIaqdz5PpOot33YbSTMtVq4nteW/e8+huay+bCZgfGaKwxAifGY9RGB+b3XLgIwTF48Hd0uoU0p4BKaVjjbBAsXn/7j4mzo7TsL2J9Z/ZiKIqjJ0adeqJeYpfP7YlySfzDHxwifEzTmTRLlhkxjJIJGa29NqbGZPREyP0vXPRGfUKSPbFwR0uWfdq4rZPtHL/z61C0UQJkboe8NW62PHZdu76QjuBBmPSOm72NgkhCNYbbH2ymZW317D/O33s/seLZGKLj0IH6t3c9slWbv9UGzWtHlRt4ZFeVVcIN3sINRmM96T58IXKCjSbeRtpSdbeU8cjv7iWlg1BFE2ZU/IqFEGg3s2tH2+laX2A1/77Gc7vHas4euT2a9S0elFdE2x+tJHHf2kd4VbPvNdAUQV1K7zc8+WVtG8N8fJ/O8nAqeqVSbsW0FwKmx9r5MGvrKa+01/W7f0yFMWJIt7/s6toXhfgjT87x8iFJA2rl78ecSn5v/8O+KYQ4neAg8BfTy7/a+BrQohzwDjwxaU10YFhCISATMYpvjxTbuP1CrJZuai+vRCdWHoh4skMN2lbCKW4U9cCQRSvb9a0/op3lUwuzntqidDDNaUldXDIUG5gkVXpSzZmkxu4hCwUpoo2T+2/tg7VH8CMTszyZRwLh0h9Cflz9FVDSyioOrmdQoHC2CieztVVTUq4EaF6vOiR+tIp1UKB7MXuBWVpKppCy91tmFmTfDyHbdrofp1CKs/4qVE6Hu5EKALbtCmkCkycG2fk8BDh1bUkB5LOZwWb9Ej5kbu0Jf27++h8ag2eiJdCuoDu1YuiYNcKiioQikDaYNkzBDPC6UBmdjBSSmxTLmhQaBXsimuh+etcPPSVNdz6idYiiwdpS6yCjVmQSFtOtU3VFVTdIYOKKgg2Gtz3s53ohsoPv9pFLln5oLCm1cOjv7iWTY804vIWl9pyTFrBMp12SOk8TooqULRiE9lC1ub0OyMVl/8x8zYNa/w89AtrnGLDikBKiZl3zsFlsqSoAtWloGrO8QshECq0bAzy2C+t5YXfztN/orL7SXMp1K3wsuH+eh7/5XXUtHomrXEkhbyFNfP8qwLNpUyRcSEEmkuw8o5aHvvldTz7m0dJji08I9rK2Yz3plF1Bd1QUVSHMCqKQKhi6jivBoQCmx5p5LFfmj5mmKy9ajvXxDKdaLNQnLZoLgXNpbD+gQYUXeGtvzhHpGP5VFSYDRURKynlW8Bbk7+fB+4ss04W+FwV2laEz3zOQNcFf/c3xS9ZtwH/7tcDfP1rac6eqfwFYE5MUI0JZjMew87lSnxzFMNA8/urR6xSKczEtReQaLWRErIDTsRvMWaLs8GMx7HSqRLzVcXrQwuF5yRWquFBKVdcWUrM6HhVooaFibFJk8iPts5KDQbLTgPKCoi0BIxaD+FVNdgFi4EP+hncPwASzj53ivYHV9J8ZyvSlgzu68fOW5z77hna7m2ndVc7VsFi+MNBpGkzcWaczFgGACtrMnpsmEK6wOCBfhRdoX5bI0IVJHrjSFtWqcTuwnF+33hZAqSogrX31tO2eXpQkkuZHH11kPjw/AM625L0HJpjMHEFNLfCvT/Vye2fasXlnX69Z+IFeg5NcGH/BINnEqSjeRRV4K9107wxwLp762nZFETVJs1wPRo7v9RBbDDLB9/uqUhz5AnqPPZL69jyRBOaXvycmHmL0e40Q2cTjFxIkRzLYVsS3VAJNLipX+kjssJHbasH3aMy0pWk71jl7zvdULj/51bRuNZ5H5g5i/5Tcc68O0r/iRjJcaekir/GRfu2MBsebKBxjR9lMqomhKB1U4g7P9fOK793inx64XITIQRrdtWx7r76KYKRTRa4eDDK+b1jU+df1RVCjQbttzj7j3RM1zFVFMGau+vY+kQze/9Xz4KjZtHBLM/+1lFcHg3dUDACOp6Qjjek07Dazx2facflWZj+qlI0rg3w4FdWF5EqgNREnnO7xzj3/iijF1JYBRvPZHtW3xVh1V0RXF6VNTsjGD6NcMvyN9JdNo51dXUq5cr42RasW6/R0KBwdvbi7rPCTCaqkp5gZdJliZVThsYzy7cqhxmPIc1rbzWrBYNT7vRF7bnsrl0l2OmU43RPTdFyxTBQy5XOmQHhdqOWqSovC3msZHWyJ61YzOm4r8676YaBHq4pO6VqZdILTpqQps25F06X/SwXy5X9LBfN0vXS2ZLlZ587VfTd4/9wZOrvvnd76Hu3p7j919gcv2vPGF17SiUFqkvBG3YVEatswuSDb/Vy6Xj1B0hbHm/izs93TJEqKSX9J+O88adnOb9vvCxBOPHmEHu+0cPOL3Vw94+vmMpgdHk17vuZToa6Epzfu7CkD0UT3P0TK9j6ZFPR1N9lgvj+Ny5y8cAEidFcWRN7RRX4Iy5q272suitCYjhHanzhEZvL8Efc+COOHjQ6mOG9v73A4ZcHSE2UbuvU2yPsf7aP+362k9s/1TalAVNUwYYHG/jwxUv0HKxMRtC8Pjj1+9DZBN//k7Oc2zNKPlV6/o+9Psj+Z/t46v/ZwJpddSiTOjhNV7j1E60cf2OI+NDC3rG2KYn2l1+3aX2AW55uuSrESnMp3PcznTSsKR7Y9p+K89ofnubCvvESzd6598fY98+9rL+/gSf+zTpq272suK3myk0vS9zwxMrnExiGwOsV6DpEIjNGQAKamxSCQYG5yErxcqnTgJOwc7myWXGKrpcIvpe0n2zmuuSpql5f2XIMdja7pGzAku3lsmWnmYSqzlt0V9F1RJlzLU1zkqwtHVYm/dHOE56E6g9CGaGzlUxU9XrfRPUQajK496c7cfuc51RKycj5FM/+1jEGTs49nZUaz/P2X55HSrj/Z1dNdb6hZoMdn21n4GSCTHz+69621YnyzCRVlmlz+OUBXvvD0yTmydSyLUl8OEd8OEf3gYVH6mZDYjTHa39wmqOvD845nThxKcOb/7OLQJ3Bpkemq2l4wy7W3l1XMbEC5/yP92V47j8co+fQ7N+XEobPJfne753iC7+3fSrKBhBZ4aVtS4gTCyRW1wttW0NseKBhSiEhpXMdX/rdE851nOXUF7I2x14fxMxbfPw3NxNurl4Q4nrihidWjz/h5uFH3WzapKNp0LlquslCQG2twvi4zYXzi4vi2BV6Gs0Ky0SW81lSFIRavdPs7OMad+xCcQqHlmuPWUBa1csokaZzHqWUJQJ24XI7E/mz1WtT1bJRFmnb2FUiA5U4wC9nKG53+fI12SyyQrf+m7j6EAqsv7+extXTnbKZs9n99W4GTy1MI2TmbQ69eIm199SxYrsTORBCsPL2WprWB7iwb+6ole5RueWpFgL104Mb25Zc2D/O9//HmXlJVbUhbcnhl/s5+dbwgjRaybEch1/pZ82uCO7JiJ+qCxrXBVB1UdF0KDjnf+83L9J7ZGGkbORCipNvDVO30ofmcq6h4dOo7/QhVIFcpAXD1YaiCjY93IgR0GboquDIKwMOIV1As7v2jHH8+0Ps/PGOipIcblTc8MTqvXfzJFMSn0/B7xccOTzdQUoJiYTNe+/mGRlZZOdepeiDtMs7ngkoyaxa4p6quK0FQhGOKL9MRytte9GFaWdDWYKKE7VyHFLLf08IBSHKnGtbViVrEZgkFTfmC66acExgy1xvy/wXEbFbbnD7NDY/2og6qWmSUjJ0Lsnpd0cqulyxoSzndo/RsS08Zc3gq3HReXvNvMSqpsXD6p2RIkKensiz+2sXiQ1e+4hLcizP4VcGFq6PkjBwKk5yNIe7w+kahRAEIo7fVjq68MGZlJKR7hQn3hhe8OvRtiQ9ByfIf74dbVL3IhRBqNlAcykUMjfmgMYIaqy8o7boumdiBY69PrBgbVgha3PyrWG2PdVMoK56MzzXCzc8sRoZsfn+azkiEYX6eoU/+ePqCaVh7vTXyjZEeSdhYNl3xHPYAjsOyqK6ne2s1+QGOI8VZmhdH1Tjnv5oZz1+1FC30kfDDB89aUPf0eiCtTmXYRUkQ2cT5NMWbv901KZpfRCXRyU/R+fetjVEqGla4yilpPvDCbo/XJwp71Ix1JVg6ExldgWJ4RzpWIHIjGVuv4ZuqEAFxMqGnoMTRPsrkyCM96bJZy1mSgS9IReqJirY+7VFpMNH+IrrPnQuwUh3ZX31cFeCsZ7UR4JYLZuY29tv5Xj1e1chlKxVh1sKRS0bLZFSzhqBWTaQ0pmiK0eeVLW66btCQcyyTVmYJ1pi204E7Uoooqw+bFG44V3XRVWc4R3zzzIRWE37yFtNLDsIaFobwB+Zzu4pZC36T8QXFUyOj2TJpaclEkI4UZPLbu3loLkUmjcEi4TRhazN+b1jZOPX1qz1MnoORSs2ObVMm1yquL2qLub0YyqHQtai71isYg+sQs4uiUxpLmVWY9cbAZF2L97wdGKTtGHobLLi655LWQxWSIRvVNzwEavLGOi3geqXWlDd1UntFLpeUkMPnA5q2Yt9pXS8vqTNlelwisvtkNMqkUfhmuU8SjmvcN+2TKRZeq6FqqLo1anV5mznxn3JCSEQVThWO5t1dG5XLFcMT1U1gzexdGguhbpOX7FXli0RqqBxTeVmi8H60neiN6ijz5FN5vKpRXYBANlEgYEF6ruuBobOLS4TuERLJUTFj3whazF2sXLn9MseZ0W7v4HHcqomCLd4ioqfWwWb0QqjVeBo/CYuZUr0tcsRy+YNGQg4N3cifsVNJyAUEiSTksXYFCleX1WmshTDKNuhyUIBe4nGlDcCzGQCaVoI1xXEyuNF0V1YVTpG1W2UzaKUhYKTkTcHZKGAXTajUCspyrxYKB7PjR2xURRUz9Iza6xkouy0p+oPIPRS242buH647Ic0szNy+zQ+9n+vx1pEtrSiiqnyIJehuZU5oza6Wy2ZwsmlTGJD1+fdJ21JcqQ6+17M027m7bLWDh81KJpjjzHz3rMtuahEBWlJ0tFCufH7ssOyIVaf+ZwHXRf85V8UM2GXC37tNwL8/d+kOXmycmalhUrdxBcD1etDMUoJgZ3LzUsIlgPM6ASykOdKMzEtEEAxDKcjrgIUnx/FKDUhsrMZrMTc+7Bz2emo1syMQl0vWz9wMVADwRt6NCVUFTWw9GM1o9GyU9iq4UEPhTGXWAbqJqoHVRN4wsVkVygCI1A9AiyUuR27VV0p2V8mbmLmro8MwirYFK7TvsHJCDQL1Z9hudEgFFHk7g8OscomFjdLU8hYWKaNUi3pxnXCDRxkLEYopBAOl0n/tmHFCo1I3eIORQ/XIMoYX1YEIdBrastGrBZCCJYDCmOjWGW8oFS/v2ypm8VCD9egekuJlZWe35jSzmYdn6UrlgtFQa+NVCXSotdGlq6zmiU6Wm4KtFIIXUcLL71ItJmIYaVLw/nC7cLd0rbk7d9E9SAUgcu4vh2RojhRrZkoZKyKy/FUC7Y9WTbmOsG25DJJdFkahBAl1x3pEMvFwDLt63rdqoUbPmIVCAg8XoHf7xiENjROX0QBtLSqhMKCfH5xF0MNBNFqaigMDy26jULTcLevKDuiK4yNVs2c8nrCSibJD6sYtHYAACAASURBVPQ7tfhmQGgans7VZM6fW/pOVBVj5aryRZQH+8t29EWwbXJDg3jXbigRq7ubW1DcBtYS9G5C03A1NC5tKlDKSWF4Karh0K/XRtCqELGyUkkKw4O4aiPF0T9Nx+hcjbJ/b1Ud929iibjinjTzNiPnkxQW2cFdiXQ0P3cESJQ+FmWTXa4VrnPffF2P/QbAYg9f2tfZzaVKkxE3PLF64kk3T3zMYP0GDU0TbN46HXUQQCAo6L1oce7c4sK+WiCIu6mFwvAwi30aVX8Ao72jZLmUkszF7o+E74+dy5K50IVv4+Zi8bJQ8KxZh7L7HezM0gikFgjiXb2mZLm0bdLnzizIiyrXexFpmiXkTK+rx9XQSGYJU5ZauAY9UrekqUBp2+WLfguBVhspXV4JhMDoWIlSJuJXKexMhmxfL961G4oiaUIIPB0rcLe1k+kqLT1z40BSNtlFiLLEfTlDSomVL37/ZWJ5vv3rRxi5UJ1STlIyp8mmtJ3pt5nQDfWGzma7iaXjcmHrIggnoWIxcOwSr989o2iVJyqUww1PrL7/eo6+Ppsv/5QXv1/wwzemRXESSCYk+/blGR9b5MhMVfGu20D69MnFjcCFwNO5Gr2mtFO0s1lyPd2La9eNBinJXOjCjEbRI3VTi4UQuJua8axYRerU8cVvXwi8a9ahXxERk1JiRifILvA85ocGMWMTuBqaiparXi/edRvJXLyw6AxGo33F0qc9bRszES+b+eKqb0AxPIuOcKo+P57Va6uWtZc5fw5zx0708BV1G70+gjt2ku3rQd6giRnStsvX1FSUqtbuvBFgm5J0rDgSq6gKuket2C18sbAKNtlEcSTWE9QX3cHexPKAtCXZZPG95+j7FvcO0twqqnadiJUAl6FWJTfphidWsZhkz/t52ttV6hsU/u5vqysEF0LgXb120SNwNRDEv3U74gpRt5SS/OAA+dHhajX1uqMwOkrqzClCO+8pIgWK2yB4591k+y4uutixXltH4NYdJecRKUmfO0NhgWJpMxYj23MRva6hyM9JKCq+DZtIHP6Q/MClituneLwEbrlt6dEOKTFjUWQ+h5hh9SGEQI/U425uIXOha1Gb9q5dj9HeUbURX35okGz3BbRbwsXlhRQF3/pNBLbdSvzAB1Vzta8mZqsPqeg6rvoGshcvXIdWXR1YBZv4cK6IrKu6INRk0Hv42rShkLNJXlEs2e3XCNS5iQ3cnDL+qMI2JamxfNG9p6hicSafAjwB3YkaXQdoLicBoxrvz2UznHjn7RyvvnJ1HlA1GCJ8z/1o4coqawtNI3jHXXg6V5dcDGmaJI8fWfL02I0EaRZIHNiLFY+VaAg8a9YRuvs+lEX4gileH+H7H8LoKCUFViJeUectzQLJE0ex0lcQcCHQ6+qpue+hitsoVI3A9tsxOldV9L3ZUBgZwSwzJakaBsEdOx1Lhwrham4hfO+DVY3G2NkMiYP7sFKlZFlxu6l95An8225devLHVYCdy2LGYyXLhcvtRPVcy9/d+TLMnM3ohVSRGahmqDStq04m7EKQT5tM9KWLVA+GX7umbbiJaw/LlEQHMsX3ni6IrPBVvC1NVwg1GxUTmyuVNkJhUeTM8GtFdS6XgmVDrIaGbM4vstDyfBBC4F27gbqnPoFe37iw77hchO6+j/Cu+1DKRKty/X2kjh+5Gs29rsgN9BPd86OS6TRF0wjvup/aR5+siKDqtRHqn/4Ege23l0xh2fk8sX17yF3qq6iN2e4LpM+eKiF/QlHwb9lG7aNPovgW9uArhofgnXdT89CjVTMZNWMT5Hp7Sj8QAt/mrQ5B8i7wxTQp+G/45OdxNTVXXZ+Q6T5P4uD+so72WjBE/TOfIvKxZ3A1NTsCiUqgKKiBAMaKTiKPP4VnVam+brGQuRy5wX7kFeZ2Qgi86zYQuOXW6+aiL+Ysf1U5pISBM3FSMyJGqiZYsT2MJ3RtSG8hazNwOlFkr+DyqqzaUYvL+9HStN1EMcZ606Rj0/eeUAVN6wIlNgzzweVd3GDAzBVnEjo+bGrFz1ig3k1t+9L1qbAMpgLngxCw824XvT0WfX2VES9p204kRFWd6Y1NW9BrI8T37yF97gxWKukYTk6SCKFpKG4Dva6e4B134d+yrezI106liO/dPa89wLKElMT378FobcO3cUvR1JhwuQjdtQt3WzuJD/eR7jqLnU5hFwrT51BVES4Xqj+AZ9Uagrff6aTvXxnxs21Sp04Q378X7Mquq53NEP/gfTztK9DrSrMYQ3fdjR6JEN//Adm+HuxMxnFslxIUxbnOhoG7uZXArXfgXbthymDUzuccnc4SojTSNEkcOYhv45YSM1Sh6YTvuR9XXT3xA/vI9fdh57IOQZDSKfmjT96HkTp8G7fg33ILWtiZrpO2jZ3Lobjd1SltUygQ2/0u7tb2spFZxeMhdOcufOs2kj53hkzXWXLDg45A3zKdF55wyuwIVUG43GihMK6GRtxNLbgaGtHrG1DcBvmR6k6bZ7rOYd2xs4ToK4bHIdceL6ljRzBTCac6gm0DYrLouOK8F1QNoTk/djaDfWUkdB5IW2KbxaRUKAL9yhT1JWK4K8ng2cTUiFsIQcumECtvq+HkD6+NHKH3SJTESG6qcxJC0HlnhPZtYbr23PQ9+6hi7GKa2EAWf+30vde41k9dp4++o6VR49lQ2+alYVXllQLSsTzWjNJBQggi7V50Q62ocPXKO2rxVmkgsuyJlaLAk0+5+cHruYqJlZ3JkDp1HP+WW8DlQigKruYWIk8+Qyg6QX54iEJ0YiqLS/V40evqcTc1Ow7UZfQ2dqFA4vABUqdOVOX4bkTYqRTjb76OYnjwrFoz1YELIUDTMDpWOpmWE+OOmDwenXKfVw0DLVyDq6EJLRRGuFyl06i2Tbb7PBNvv4FVZjpnIchd6iX6/ntEHn+qLHnxrtuI0b6S/MgQ+eEhx//KshwfKH8Avb4BV10DiseDUBSn1EQ+T3T3u3g6V+NZ0bmodl1Gtvs86a4z+DZsLtaCCYFwufFt3obRuYb88CCF4SGsTNppn6qh+v246urRI3WoXt9U1p6UkvzIMMlDBwjtug8tEFxSGy/DjEUZe/0VGj7xGVyNzSXtRVXRaiMEd+zEv+1Wx7stmcBKp6cyNIWuo7gNVL8PRXc7JaD06ugZZkN+aID0uTMEbttR0mYtEKT24ccJbNtObnAAKx7HLuQRk8RVuFxOe71eVJ8fxW0Qfe8th+hXANuSZFOWU2JmMkPO5VEJNlanEsBl5JImx14bpHNHLZruHKvh19j54yvoOxYjUSUX8rkwdjHNhQPj1LR5pq5roM7Nrp9YwdC5JMlFuHHfxI2PTKzAhQPjtGyaNk/2hl1sfaKZ/hPxBdVLVHXB+gfq8ddVPisQG8w6BCo8vax1S5hAnZvx3oUNhHy1LrY+3lQ1fdcNTayCIYG0IZGQGAYYntKD1lRBfb266Kh+/MAH2Pk8wTvuBE2f6thcDU0lmWVzQUoJUpI6dZzoe29/JLyr5kJ+cICxV18i8sTTjmblSnLgduNuasbd1LzgbV6eusv2XGT0ey+SH+hfdPukaRL/8AP0+npCO+52IlFXCLBVnw+PbxWelXNrp6SUYNskjxwk9qO30QJBjI6VSyIFdjZL9J03cdU3otfVl2xLKAqa34/mXwMLmCKTUmIlE0Tf/SGZ7vMEbr0dqkSswLGxGHnpeSKPPzUpkFeK/a2EY2SkejyoHg96zdJNSpcKO5sltm8PRsdK9PqG0mibyzE7XYjhqV0olCZWLAQS4oNZCjl7qkCxy6dNRZIqGVHPuRsbTr09zB2faaNta8h5BhXBqh21PPCvV/HWX3SRHKusxIpQmIyACeJD8+tbc2mTwy/1s+GBBnw1zrlSVMHae+p56BdW88afnSU9UZmPnFAFQsxt9XAT1xe2JTn5xhC3f7INI6A5956AbU81c+rtYboPjM9bDLx9W5htH2tGc1U+bZwczzN2MU2oaVqfVb/Kx6aHG3j/Hy/OW9ZJcyvc8ek22raGqzbQu2E1Vm43/L+/GeBX/p0fRYHPf9HLP36ztuTnH/6xhnvuXZz2RbhcYFuMv/Easd3vLY0MWRbJo4cYffn5j+YUYBnk+vsYfu5bk+Q0t2RTPFnIkzx6mOHvfGNSV7XE7eVyjH//VWIf7F6SmaXM54l/uI+x11/GSqUojI1Wpeh0treH0VdeID80sORzZ8ZjjL32EskjB7ES8bLC7SVBSrIXuhj+zjeJ799b1oX/RkSur4fRV7/rlGS6Tn5y/afiRRoURRFsfqyJbU82oVbRjiAxkuPtv+wiHZ0mL6qucOfnOvj0b29l3X318051uDwqtR1e1u6q48l/u4Gf/vMdbH2iaWEp6BK6D0xw6LuXiqIUmkvhzs+187n/so0ND9Tjq3XNqX9x+7WpjvETv7mJzY8ucP83cd3QezTGmfdGpoTkQgiCDQbP/NpG1t1bX+rOPglVF6zZFeGZX9tI3SIE7+BEzLr2jhXdcy6Pxj0/2cn2j7eiz1KVQAgIt3h44OdXc//Pr6qqNcgNG7EyTfjey1ksy5GW1NYq9HRbvPq94g5SVeHLP7k4wZnQNITLjZ1OMf7ma+T6ewnf8yDullZHX7GAp1laFoXxMeL79xLf935588ePMMzoBKPffY5M11lCO+/B3dxadnpvLti5HNlLvcT37yV18lhVvZHsTJqxV18i19c73b4Flo6RlkV+eJDY3t0kDn841a786DB2IY+61BI0UpI+cwozFiV8zwP4Nm5G8XgXfO6klMh8nszFC0y89X2yl81oFYX80CDeNeuX1r4yKIyOMPrSCySPHSW08x6MjpWoPt+iR3pSSuxMhvzQVbImkZL0qRMMZbPUPvw4RvuKiu/PpWK4K8mFfeNsf6Zlar/+iIunfnUjq3fWcfqdYaL9GayCRNUV3H4NX42OP+LGE9I5/v1BLh2PL2hfp98d4a2vdvHgV1ZPRY00l8L6Bxro3FHLcFeSgdMJxnpS5JKmUw7Ho+IJ6tS0egg2GoSaDIKNxtSUou5ZeBTBzNu8/dfniazwse7e+qnCzaqusP7+BjrvqGXkQoqhc0lGu1NThNNlqHjCLmrbPISbnR9/nQtVUxg+n3KI2M2g1axQNMdew2WoqLrjYWb4Ndw+DZfX+T3S4Z2Kml5G+y1h7v/ZTrJxk1zaJJc0yaWcHzNnY5lO3b/48NzvZDNn887fXKBpXYCG1f6pqFXzhiCf/S/b6No7xrn3Rxm9kMLM2xgBjboVPjp31LL6rgiekI5tSS4diRLp8Dnke4GwTcmJN4bY+kQTzRucKL0QEGx088yvb2TTww2ceW+U0YvOPa+5FMItHjpuCdN5Ry11nT5UTSGbNLl0PEbHLeFZydhCccMSK8uCd952HjopIZWy2bsnz4svlBKru+9ZXIqkEGIqPV0WCiSPHSHb24Nv01Z8GzY5ho1uA6Fr0xlPtu145ORzFMbGSJ89SerUCfLDQ1WJYlyGnc06kZErYKVSN9wLRpoFkkcPkek+j6dz9aSf0gpH/6Prjhbt8lTh5PmThQJWOkW29yLpM6fIdJ93CjlfhaiCLBRIHPqQTPcFfOs34Nu0FVd9oyPw1jRQFUAUX9vxMVKnjpM6ccy5DjOy4gojwxRGhrF900JLMxZbnJ/TpN/Z6EvPkTx2GN/GLQ5Z8ftRdJfTvsvnTkqkZSELeax0mtylXtKnT5I+d7rYP0xKcpf6iu4fiZy/JNBCm2wWyJw7Ta7vIu7WDjydq5zptkjd5DmdvOaT04NMTpNL20ZaJrJgYueyjr6t/xLZvh6yvT3YV7FYebb7PEPf/ie8a9fhXbcRd2s7qseD0F3TbYVJi3HbOc+miV3IY8Vj5Ab6yfWVyeRcAHIpk33f7qVjew2RGcJuI6Cx7almNj/a6NTUk0yWhhE4M60CM/f/s/feQXZl953f59z0cuoc0ehGaITBBEzgDAnOiFkSFShSWkm1a1OyVdq1bK/XW1tWOexWOdSuZXtry3ZR1kr2llYqrVcyRWkliuRwyeGQMwQmYQZhkBpodKNz9+uX403n+I/7+nU/dCMDMwNyvlUzhdfv3nvOue/ce77nF74/n6UL5dsmVr6rePOr8yipOPZr4233iBAQihmMHEkxfDgVtCdVy3IUtKdpot3+vaC67vDN//UivivZf6y3Y5GyogZDh5IMHkyi/C31/MT97cOPG5J9Ib70Px6hdzyGEdbRdNH+3bfOqQ2iu4HhwymGDiRbj2gwB5VSgZK+J3GbkqlXsvzlf//uLd15K5fKfO/3pvnsP9hPZngzzi6aNnnkswMc+kQf0lcbOThoukA3NIQmkL5i+rUcJ/9igZ/8h5PEuDMv1Pq1Gq/+6xk+/9sHiaaDc4UQhKIGB17oY9/HeoK2Zeu1pAl0U0O0Xq2eIzn5tXmmXlnnF//Zoz+6xAo619iXX7Jxve3rrpTw7lmXQuHuFmRta2Helsp36bVXqZ5+GyPThdXT2w6yRohgUatUcXJZ3PW1QC/pPhKqDdQunt9RbVza9g1rzb2vUAq/XKJ6+m1qF8+1A8DNdAYtGkMzzOCBdR38WhWvWMDNrePVqu+NereSeIUcpddPUD17GqOrG6unBz2RCsiAEEjHwa9WcHLruOvZgMTukJHoZNdY/pM/7IgrU553T8RF2jb1SxdozEyjx+KYXT2YmQx6PI6wgv4pz8NvNPBaiRV+tRwkBVz/UChF9dyZ64RGFdK+sxibW/a52aQxPUVjdhotFMZIJDC7etATSfRoNCDVmrZJUpzgt/fLJdxSEdloBFmWD+D52Ql+pUzlnZPUzr+Lnkhi9fZhpDJokUj7PbBBrP16Hb8azNMgO9gOMgfvBirImHvpK5f5zH+xn1R/GKFtLnzaTV7ivifvWJnBqfu8+dV5CksNjn15nJEjKQxLa7UnEPr2BXYnSD+wVtSLzh3v5bIzNf7mdy6ydrXG4z8zRKo/3G6zveDfotyNUgrpt7IqP2CbyQ8adEMjNRAm0XtnSRGaJm7+O6S4bW0nJeH8S6sIAS/85h76JuIdBE8L7TzPPdtn6ofrvPS7VzAj+l255KSnOPviConeMMe+vJtYxmoniwhN3DB2SymFXfM4841lfvD/zKBbGtWcfXcCp1vwgSZWW3H58s4vXynhq3/WuOsizDtGvUsZLAC16l3vUu8VynXwSvd3IXyvoGwb17Z3tLi971Bbftv5a3d3DSnxK7dnQbhTKMfBc/J4hTz3EsWkXPe9i/XzfWS9hlOv4ayuvDdt3i2UQjabgUX4Pss73Ay+qzj9zWVqRZeP/u0xRh9P31rnRwXZfk7zzomn25Rc+N4ayxcrPPpTgxz8ZB/9exOBvs8t4NR98gt1Fs6WOPedVebPFO+K2BSXGrz8L68w9UqWoz8/zMQzXaQHI7eVebXRh2tvF5h+PXdrQ/aGxWXj470QMaWuu9ZtXOy69u+FCKq7Hcv1fbhfuINLerbk7Isr5ObqfPTvjLHvWCuub4efXMlgjpz+myVe/7N5yqtN9n60B928u1gnz5Yc/+NZ8gt1nv7SCCOP3vwZ8+xAe+2dv1rkzDeWaZRdIimT/Hyjrae1YcW7U4gPQhVuIcQD7YQwTXp+9ouknvrItu9W//zfUjn5BpYZIxLqwpcu1foq79UWSddMYtE+NLF1Aigq9VV8/+aWnHAoTdjqrF3neg1qjTtZMARhK4lpxqg1skh5l7vyB4idxun5zdbv9CE+xMOFcMKgdyLOyCMpBvYnSPaHMUMaSiqchk8151BYrLM+UyM3Xyc3V8ep35tVL5ox6d4VY3AyQf++BKmBMKFo8M5xGz6NqkthoUFursb6bI3iSpPqun1L98/tQtMF6cEwPeNxBiYTdO+KkugJBS4XpXCbkmbVo7jUYP1ajezVGqWVBpXb7ENmOEKib9PKoCSsTlVw7jTrUkDfRJxwcvN97DUla9NVPGfnjmi6oHtXlEh60/vhNiVrV6rbClPfCrop6NsT74hra5RcctfqN5UtMEIafXviNwwSvxc0Si7Zq3dujd/o067H0wxOJoO4PUvDqXvkFxssnC2xcKZIfqHe/o0PfqKPv/U/P0aoRYg82+el37vCy79/9Y7aDicMesdbz9hkgmRfGDOs4XuKetEhO13l2qkiq5crQSmmDY+0FuhpxboDd6L0FOszNZrVHb1EJ5VST+049jvq7fuIoWGddFowdcnDewCesO70PibHf4ZqfZV3zv9rfPneWIvCoQyH9vwCkVAGTTMAgUJy8ty/olS5sbVMCJ3B3icYG/wYmqYjRBBsv16Y4tTFP77t9i0zyoGJnyedGOXC1b9iNXf2Pozq/kGgMdDzGLuHjqFpRnuc+dI0b5//w/e7ez/WMMI68b4IuqXj1Fxqa432y1/ogtRwjFq2iRk1iKQtpK+oZZs4tU7yboR04v3BdeyKQ229iboN7ZuHFc2Kx/zpIvOn37vs4XrBpV54b9vcCukr8gsN8gsNpl7J3vfrFxYbFBbvQ6aqCpINtkLvShH75DH0TAplO9TeOos7t4S5a4jYU0cQlkl9cZX1t84idJ34x5/CL1VI/MIAeD7VV9/CW8shQhbhw/sI7duNEBr2zDz1k+8iNEH0ySNYY8PIZpPcO+dxLi2hJWLEnn0CT66T/MIYyvepfPc4ofFR9EwKLRJCz6Rxri1Qe/MMS+cfjBX9buHZkqXz5Tvql9DEfdEqaFY85s8UA2vrHUBJyM0Fm5l7wUNDrD7zmRBPf8Tiv/ntEuXyj85Lt+kUmZr9JpYZIxxKMzrwLKZ56yxHpXxW189Sq69h6CF6uw7Qk7nzLDAhdEwjjKYZGPoHr36aQrKWe5d6I4uuh+lJ76Ov+9D73a0fe8R6Izz1awfoO5hBMzTchsfM95c4+7VpvKZPKG7y6X/yNHOvrdI7mSbeF0EzNdYuFDjxlbPU84E1Nt4f4fFf2cfgoz1opoZTc7n87+e58PVZ/BtYCH7UoGNiCBNXNZFsjlmgYYkwvvLwcACBJUJIJfFwMbHQhY5C4SkXn+07Th0DQ5gItCCBQbl47GyV3n6s1zr2+vetwMREEwYCgcTHV15H+7c7JoOgPR8PU1hIJXGxW32xkMrD5f0LiRCRMKnPfxK/Vqf+9jmEoSNrdbRkgtTnP0Hz3Sm8fJHo44fQQhb1d84RPXqY5oWrNM9OET64h8RPPEvhq98k8vhBok8eoXbibVTDRjoOQkD06COE9o1Re+MM5kAvyc89T+HP/gZhGkSfOkLj1Hma5y6jUCjHxejrJvrUEaovv4a7uEri0x/DXc7izMy/b/fpQ3TioSFWiaTAthX1+o8OqQLwfYd8KQgyDltpBnoevS1iBVBvrlNvBnFMlhmjO73vjtt33BqzS68Si/SyXpy64/PfC9SbOerNoCSGoVv0dh14n3v04w3N1Hjsl/fSuz/Nid89S2W1wcAjXTz15QNUVupc+d4CCAjFLfZ/dpQT/9e7rF8pMXikm4/+p0eYf2ONqRfnMMI6R760h4FHuzn+u2eprjUY//gQj/3yPkrzVebffO9ioN5PDBq7GTL2cMl5i5LcjEuMaUkOWs+S9eeZdc9hYjFpPUVDVqnIAoPGBGEthlKSoswy716kroLi3gJBSutl0BgnrmUwMJH41FWFRfcyedkZC5fQuhgx9pPQ0ujCRCpJU1VZ8qZZ9xdRLXKloTNg7KZXHyUiYgih4SmHsswx705RV4F1YsDYzbCxh0vOSUpy00IVFQkOhZ4j6y8w677LgLGblNZLWeYYMMZxlc2s+y5d+iC9+ghNVWPKOUlDbS8E/l7A6Mmgd6cof+v7eNl8+++hyQmEYVA9/naQfKFpxD/2JM1zl5GNJvVT53CuzqM8j+RnjiFCFqGJXdgXp2m8s1mVQ4Qswo/sQ7keZl83WiQUtNmVQlZqKNuhceYi7kLn7+XML1F/5zzKdog+/Sh6+sNi1/cVQkMIDSXvzj32gRUIvR6zsz6WBdHYh2m49xNK+WTz55ld/D6288EyJX+IDybivREmXhjm3L+bYeFkltJ8lcvfWWD5dI7Jn9qFbmy8VhQLJ7NMv7xIab7KzCtLFOcrdE8EWjOxnjC7jw0y9a05FlvXmXpxjmbRZvex21fsf9hhCIuQiKHRGVgu0AiLKCZb0sdFlEFjnN3mYaqyyKJ7mZosMmRMMG4eQWx5pce0JBGRIO8vM+9dIu+vkNJ62G89SVhsbt4MLPZbR0nrvaz7i8y5F1nz5xAIjOvS3rv0AfaZTwCw6E0z716iLPOERBSxJULZECbhHcakidaYhNVuu1sfJKX3kvOXSGnd7LeeIiLirPtLpPU++oxd9+Eu3x2EYYAvt2WECkMPMqdaGa3KcYNEKE0EciK1wC2pfLlZK9MwkM3r4mY1gbAsVCtjV9YaVF5+HT8fCPwq121faytktY7y/HYbQjw0S/lDgXj/ON37n7nr8x8ai9Ubrzs8+aTJf/wbMb71zSbNZqflamVZ0mj8aFmzPsSPD4RukB46SNfIEcxwHN93KK9eYe3yCdQdFqF+0Ih2hzFDOqWFattLpDxJfrbMoSPj6K20aqWgMFtuHyN9hVv32mrjsZ4Isa4wR760h72fGgUCocPMrgS13I+X0O6dwBAhppy3WfWDrNZlLITQyOj9hEWUhqqiUCx7M6x4s1tcdIKGqjJpPUVUpGiqII4kJCJERZIlb5oZ9xyq5brbIGlqiyswoWVQKObc8xTkpkVRQ0dy9/N02Zsm5y8T19IktW7OuyeoqwoprYeElkEgOvrxXsHLFUAqIo9M0jh/GaHryEYTd3UddJ3Q5AR+vkj4wATeWg7ZbAVCX5cUpmwHd3mN8KG92DMLAZESAlmuYl+exejJ0Dx/BaUUwjTwqzWMzEbCzg7j/nCpe6BoFJZxqvlbH3gDPDTE6mPHLI48arJre74Z1gAAIABJREFUTOfnvxCm0eh8zP7Jf1vmrTfvQ0abUmiaQTIxQldynHAojVKSan2VXPEy9cb6Dg+4wDSiJGIDpBKjhENpdM3E922qjSz54jT15jrqfqXY3CNMI8r4yE8QsraYjxVcW/4h5erCtuNHB54lERtkcfUtStXAjy+EznDfk2RS49TqWWYWv49SwYs1Eu5m99Axao015ldeb49bCI1YpI+u1ASxSC+aZuI4FQrlGYrVeTzv4SiTcv8hyAwdYvyZX6RWWKCWX0TT9CB9+m5ER+8QuhEiObifanYGt3lrl4tv+ygU+tYMJCEwIwae4weB562U+pvVePNdiedIpl9eYu1C50usmr1/c0HXLCwrjiYMlJK4Xh3X2x6cahgRDD1E0y6x08plGhEsM9EWPnTcKo57f0RX7wQ1WaK4xb3m41KVJbr0QXRhbhJZfDR0QiKKjt4iSsGXuti0JDVVjbLMMWRMINDI+gvUZKkV19WJgr/GiLGfvdYTLHlXKfgrNFTtnkiVxKchg/toqwausrFVI4gdw0EnSOp5P9iErNQo/fV3iT//DKHJiUBv7vXT2JdnqXznVWLPHUXoGt56gcp3fhhoIeYKm9Ykxwk+S0nttXdA00h9/hMgfeyZBWo/PEnt+Enix54m9XOfQkmJu7hK9ZU3UZ6Ht14IrF5b+9S2VgX3wy+UkI0fk42IEJjhBHoogu82cevlIE44msCtl4NNqBCYkSS+20S6NroVwYwkkNLHrZdQvhfUYg0n8F0bMxIIPTvVAkpJrGgKzQrj23cfwP7QEKt3z7r8we/f+CU2O3OfdvVCMDrwEQb7jrYlEDRNp7/7CAM9jzI9911ypStsfcgtM86+sc/Rnd4XCDkqiVIKTWgMaAaNvie5uvAy2fz5Dwy5MvQQ4VAG0wgTspJowmA19y5lthMr04zR23WIan21TaxMI0Jf92EyyXFqkSyLa2+1XYmxSA8DPY8yt3yifY0gi/ExxoaOYZlxlPJb1Vd0BvueYL1widnFV9oxYz9O0HSDRN8Evttk7u2/plHObllHHvxiEop3M3jgeWarudsiVpWVOqXFGqNP9bN8OofX9AmnLAYf62HlbA7f9TtJ142us1qnOF9BNzXm31jDbXpousAIG3j2/XmeI6EMY0PHSCV2oWk6Uvqs5c8xu/hKeyMAAenfNfAcmdQ45658jaZd2HatTHKC3cMfR9dDmEaEa0s/ZG75h+/5M+0qG6k6Yz8UssMVJxAktR4GjXGiIokuDDQ0DGG13HObx/p4XHbeYdTcT68+TL8xRlUWWPFmyPqLHaSpJNe55LzJoDHBuHmYUWM/OX+ZZW+GqipyN/NVodpWsmAkcst1FO9rxIpSOHNLFP6/byB0PXi328EGvnlhGvvqfCCC67oo1wMhKH7txcA1CLhLaxT/4tsgJbJap/Ld44iQGVjgWucox6X83R+iWWYwajeoTuDbDsWvfavd3gZqJ88Gv16LcJW/9QPUeySy+34jMbiPzPijKCnRzRC56ZM0i2sMPv4Z1s6/SiO/hBlJMHT0s2QvnMCza/Q98jxCM9B0g0Z+iezF1zDCUYaf+jzNwgpGJA4KVt/9Pp5dI9o7Smb3ozi1Motv/vVd9fOhIVbTV3ymr/h3VTXkThCL9BKykuSLV1gvXsZ1a0TCGUb6nyEZH2Z06Dkq9WUcd3MBktLB9+3A8lK+Rr25ju87hKwEg71P0J3ey9jgRylXF2ja73+BZtdrcPnat9A0E8uMMzn+eVLx0RseX62vIoBIuIuNFX8ji9F2yoSsBGEr2SJWoi0dUa2vthYdQU9mP3tGP40vXeaWT1CuLuD7LtFID8N9T9LffQQQTM1+A89/H3ZfQgviIFoLjlJyZxec0AJrEhulH7zrvt78TtP0oIyL8hGa3gqG9DcXYiEQmo5uhjEjCTy3iec0WvESCqm2ty+EtqWNm/dR+i5BuZJAokKhguPblVKDMce7d2GGk2iGhaZvxNSo1vnbYVcczvzZFZ75jUMgAqLVdyBDJGVx4uuzN7VSbUWjYHPuL2d48suTGBGd0nwVK2aSHIxy9mtXWT1396Z4CMh8f88R+nuOsLj6FuXqAppmYjulDlK1AU3T0YR2Q6XzXOkK1foqidggB8Z/Bl27eUHjO+rrDn/TMTrI0iZu7RSLaxkOWE8DsOBNUZUlPOWQ0ns5aG2PHamrMpedd4iKBF36AL36CJPW05humEXvSpv4KCRr/jwFf424lqbXGKHP2EVa7+WS8yZl2fmbXd977YZjun6EDw4aQQmrG1nZArfmVnIXuPK29UkpVNPu/LtS7XgpICiPtPWz76Pq/vZreT7Su64/SqGaO2REul7H+cq5naxJ0Rp365wOMvtwQA9F6d77JOXFKSor08QHJuidfI6541/FqRZIDO2jkV8i0jWEUuDUi3TvfRrpOqxf+gFWLM3A45+msnQF32lgxdJUlqZYv/R6kAXrNEFJStfOoRkhYn13H9v30BCrjz9vMbHH4Mplj8VFn9VVSa16/x8/Qw+znD3F9Nx32lpWxco1mnaJI/t/mWRsiGi4u4NYeb7N9Px3kNLfpn9Vra8RjfQQDXcTj/R9IIgVKFyvATRQyr+lKGi9sY4vHcKhNIYewvObxCJ9aEIjV7xMX/dhopFeStUFNM0gFunF820advCSDZlxdg08h65bTM1+k7X8uTa5KFZmqTXWeGTvl+jtOsBa/hzrhUsP+gZ0wIqk6Bp7jGTfBEYohvRdmpUc2enXqeU3LXhWNEXP7idJ9I2jG2HseoHctVOUli+hZECeBg68AEoifZf08CFquTmy02/Qt/dZYt27KC1fYvXyCaRnkxk6RM/4k5iRJJFkH0I32P/Cr6NkcP7i2W9TXr3Sal0QSfXRPfYE8Z4xhNCol1bIXTtFdf0aW1UUUwN76dvzLHOnvkEsM0Rm9AhWNI1n11idepXy6jRC0+kZf5L04CSx7l2Y4TgTz/wSvhfMX7u6zswbf47vbie5SsHMK0s4NY99nxkhM5agtFDlpX92kuxUML+lK1k6tU5lZdOcrnzF6oUC1bVG+/P09xaprjXY95kRRp7qw6m6rF0qUF66dxebrpkkYoM07SJzy8c7ntntY5JcW/ohi2snadzgGfV9m7pvI4S2I+m9W0jlB4HiojNQPKolrhMOvn2ktV5iWpLz9mus+LPtvydVd0eAe0c/8KmqIlWvyKp/jUdDLzBoTLDizW5zC7rYFOQqRWeNop7lcOg5uvShNrG64ZjE3Y/pfkAg2Gs8jilCTLlv49IZTB4VCfYaj7Psz5CV2633DyN0dPq1Mfr00bbURU6ucM2/8FCRKyMUJdI1iGZaJIf3IfTNjU1l5Sq9k8+ih6Ikhyeprs4gXYdozzBGKIYVS4HQ0TQDzQzhOw18p0ktO49nd75rgg3rvYllPjTEyvPgwEGDz3wmjELhOpDLSaamXM6f83jnbZfqfSBarldnOXtqG0GqNbLUmzni0X7CoRRUrj9v55iQplOkaRew4iOYZuye+/d+wHFr1JsFIqE0hhHB85sk48NIJVkvTpFJjZOMD7OyfhpdM4hF+7CdMk07cA0mE6PEYwPU6musFy5tc51U66uUq0v0dR+iK7XnPSVWkdQAu5/+IuFED9XsLNX1a+hmmEiqH8PazJwKxbsYf+aXsKJpSssXcZs1YplBxp/5RZYvfJ/VqR8GEgOxNKmB/ZRXr+DZNfr3fZRY1yhOo4xn1+mfPEajtEpx6QJ2vUBh8TyaYdI7/hRmOEH2yhv4voOSPs1qrt1+sn8Pu47+LCAoLV1ASp9k3wTpZ3+FxbMvsj77dvtYw4oS7xlj6NBPEEn10yitUsvPEU70teobKkDh1IsUli6g6QZazxi5udPYtcAF5jt15E1qUkpPMf/GKvNv7Kx+79Q8Xv6dzT6NPD/Grk+NYyVCVBrLW64jWT69zvLpwAWc2d/NxOf3Icwl4CaVBwSMfHyMvicHeef/eL1DUNQ0olhmnJCVIGQlAUE82ofnB/GSjWa+bRUVQicW6UFrWZ+k9ALL4l0SJ9OIYhpRbKfcis/0adhFNM0gEsrgS4dGs8BWa0hNltCETr+xi7os42ITFUkGjQl07q4YrI+HQhHREph+oE8X19IMm3u32YsiIk5cS7fiqlxQYIkIOno71mkD3foQnnKwVSMgTyLQpQp0rzbfmTVZRgiNPmMXNVlqjSnB0D2M6f5AEBUJLBHpsOBsIESEjNZLWeZZZ/E2bIMffPTqo0waT1GQq2T9RTQ0XOz3hFQZmPRqI+TlCvY9FekC6bu49RK5qTeo5xaBwDPg2XUa+UWU8kmNHMCKpVg79wpKSbxmnXp2nvzVU2w8c55dxwzH4UYW//uAh4ZYHf+hw2snHCxLsHu3zuQBgwMHDT77uTC/9Z8Z/L3fLHLih/cuJGc7ZWyntO3vSkmk9ACx445LCJ1IKE08Nkg03IVpRNE1E123iEV6W8c8nCmxnt+g0cwR7zqIZcZw3RrxaD+2U6ZaX8W2yyRiA2iaiWnGCVlJ6o113FZgbyzS3YrpSnFg4ue2XV8IQSI+RFBeJ7Xt+wcFzbDo3/8xIsleZt74GsWlC63gR9B0s/3QCc2gd+IZIql+po//v5TXpoMkB8Nk5MjnGJg8Ri03RzUfxJ8ppVg6910UEEn2YYRiTL/2bzFDMfYd+w+JpAcoLl2gXliiXlhCN8MkeycQms76tXfwnc6gScOKMjD5cVBw9fU/pV5YAiAbTbLr6M8zeOiTlNdmcOqFjnPCiR6uvvZnLYKmEEJvW7aU9CktBwQ2kuglkhqgsHieev7B7NLXz2Vxqg6Tf+swyd3pGx6nh3RC6TDiFkWChRBE+2J0Tfa03Zwb6O06xHDfUQwjQthKghAcmPj5oHRKyw1erAQZdaYRZffwC8Sj/VhmDMetceriH9+1ZbkrvZfR/mcolGfo7TqEEBrzyyeIRrrpSU+ilM+l2W+0tesgiFta9mbo03eRDPfgKRuBRlGuERKRu+pH3l8h768wakzSow8hWzFYBX+NiIh3HGuJCPutJ9tCowCmsPCVxzX3PP4WQdEefYg+fReucpB4LcHPEFlvnqy/OXfKrTH162Mkw92dY1K3p9P3fqCs8rzrnqAqiz8SpEpDJyP6cJXNJe9tGqpy65PuI6IiwS7jAFW3hK3ujVi59QrlhUukx45gxtIIoeHWyxTnzuG7NqXFKXoPPEtleToIUpc++atv03fo40jpo/wgBq5w9dQN29DMENGuYaLdI4TiXSSG9tMoLOM17uy+PTTEqrtH4+BBgz17DHaN6QwO6ZgmXL7s8dJ3bWZn7k+dG1+6+HdoBjTNGMO9RxnofRzTiKKUh++7+NJBKtneDT+skNKj3swhEETDXbhunUg4Q6E0g+vWqdZX6clMEraSrYxIi2p9LQioFRqGHmYjc7I7vfeG7bhe/T2tVWiG4iR6x6nm5lukqvW7K5DeJknXzTDJ/r00SqtUsjNtciI9h/z8GXr3PEO8Z4xaIdhF+U4Dp1FGN0K4zSpKenjNKppu4nsOutGpcN9ZvHX7y9yKpoh1DZOff5dGcZmNnZdTL1FcukB6cJJoeqCDWCEEubnTNCvr7eOV2nletxeQB1g3tJmr41Zt6qtjGDHrhsflzmd58385jtu4+3mwXrhIuTqPoUfYs+tT6JrFpZmv40sHpWQr6y+A41a5NPPXGEaE8eGfIJUYZeeIp9uDQBAJZ6g3c0zPf5fx4RcYG/oY2cIUl+e+zZ7RT9HbdZBiZa491z1cpp3T5PQloiIJKMqyQE0WKes5HK1Jz5E+Bh4bIR2KUZzJI16FDe+cQlHw11BK4bQWr6aqccl5qy3B4CmXklynLss0VY2a3LwHFZnnvP0aMS2NKUwUCkc1KfrZtuDoBmbd8xT9bJBpKAyk8qipMiV/vcOt5uFy1TlDXl9uj6lmlkkdS1DJr5E7E2wOCnINz3VxW9audW+RilZAEmSernjX7klqQSDQMdHQkMgbxlXpGK3sQ6jIwg3V6TuPD5IAgmglf0flewhiunTM9jiCo71tY9LQ23F1ErlN9d7AbJ2v0DHabW70xcNrjy9o02hppEXw8dAQWIQB8HA6FPE3xqShIxD4+B2E+vo+ujgIaN/bIIPTbVvBguN0Mlo/lggF/6mgbYXa5oLdgOf4VNdt7EhwL31HYm/UyVSS9ctvEu/bTSjZjfJdmuVs672lqC5dwQhFqSxdbm+Ka2vXWPV9oj0jIAR2OYv0XFCK/NV38JqdbkDNsAilenAbFbxmjXCqF7de+tElVl/4Qpgv/3qUuTmfV19x+PaLNosLPsVSEGt106B2Bfg+0t3hYdnpxDtYYDRhMNL/NGNDH8d1a8wtH6dYmcVxakjlo2smB/d8gURs4Lav+UFEvbGOlB7RcDdNu4RlBEWbPd+mXFtmsO8JIqFMQKx0k2q95e5RbImnmuPq/Hdv2s6NXKoPAkYohm6GaFbWb+pT180QRihGNTe37TjPruO7TaxIqh1ULqUXBKkTWIZ8rzXvNrL8xJ0t3EYohtBNnEZpmxvVbZSR0sOKXW8FUtiVHNzlgtQBAYmRJOm9XUFdwKpDeaZIdWnzZWPGLboO9BDpjtAsNilcymEXbz8JwYyZDDw9jBE18W2P5dcXcaudFmgzbtHzSB9m3KK2XEEYO1uAAxmEKqYRxfdtBIJqY+0GRc0DK5bn2/dx7gmyhYtk8xfIJMfo636E1dxZSpU5+rsPEwml0YSO3LJwudgdFp8NLPszpMYyPPtfPk91qUIhv0I5XUSKToJQktkOhXMIyNWyt7147aJ3ueOzxKeeLNHQyjTzN78Htqq39bNuhevHZMVCTH5yktyFLPlTKzv2OyeX2brer/k3rpd6K2joDOhj9GtjWCKMrRqs+DM7Bs8P6hMM63vQ0NEQXPXeZVnObDtOIEhrfQxou0loaTR0POVSU0UW/WnKqjN4PyqSDOnjpERvULIHn6aqMeddoqg2xx0TSYb1vSS1bnQMbNVgVc6x5s+1SJHGmHEw2PQh6dGHqcgcC/4VRvR9JLVu8nKZWe8CHg692ghD+h5CIkxEJNDQeMx8PnDZ4nHFO91W3xcIkqKbQX2cpNaFQFBTZVb9a+TkcgcB69IG2KXvZ8p7h7TWQ682QkhEcHGZ9c6Tk0to6Izoe+nWhkhqGUxCHDSexm+RvrqqcM49sSN5nT1Z4A//7ltBzUCCTWe9uHmcdG3Ki5dgcfvv7TbKZM+/2vlHpaivz1Nf7yz340uPwszpbdfwGhVyU29sv/gd4qEhVufOebzyA4d0WuOpp0wm9hicP+dy6WIQzL606LMTbwJQvkfhBy9RPvn6tu/c/L1lHllWnP7uR9A1g6nFl1lae4eti5lpRG4rC+aDjlpjHV8GWXy2W8WXLrX6GqCoN7IopYhF+1pSCpJaI4iZUUhstxpkxgmdcm2p5VL9IKBVMPhWv09rRyR2IkQCEOK6eIXrycy9kZtNHbCd+imCnfB1hEttOe9e0XWgh6P/4Fmcso30JGbUJH8px5nfewslFdH+GEd+4yjx4SR2oYmVDNEsNDj7ByepzN+emr/QNeLDCXqO9JPek6F0tUBpC7Ey4xaP/SdP0X2wl9pyBYTAit/Y8vV+QimJ3YovdL0GntfEcQIS6vk2lhm7fXItgvsvdMGbv/MqTsV5IJJOuz+3B7vUZPbF6QeWkudWHE595Q28xnvz/A/ou9lvHKUsc6z4s+gYjBj7iYkEDdXpbs/681RlgZTWw4RxpKUMv/1GJ0SGA8ZT+His+Qv4eEGMmkhiilDH4QnRxSHzGUIiQk6uUPBXMYRJTKQ6EgiiIsFh81l0TLJyAUfZJLQM+4zHiYgos94FACzCdOtDZOUCNVlkQB8nLtJUVYmaLDGk76Ekc2TlAlVVYtGfRhc6I/o+QkSY9c/jqcBSVlObFssubZBJ4ygeHll/HomkS+vngPk0V713WfSvtI81MEloGXYbhwiLKGWZpyzzRLVkO6FDIamoAk2/gWScjNbLgn+FeqsskYfbJlnXw6n791wA+YOAh4ZYnTjucOK4QyIhGBnVOXTY4OiTFj/3hTDJhMZ/9Y9KnHzrRsxK4RXyeIV7I1E7wdDDGEYEX3pUaitc/yBGQl1Ewpn73u57Dcet0bCLhKwEqfgojlen1gh2XE2nhO2UScZH0DSDpl3syMCqVBex3SqxSA+p+CiF8vad4PsBt1nFs+tEM4NoRgjp3cA87Tawq3nC8W50M9yRKReKZjDMCM1KLvDhP4h+Nip4dp1wohdNNzdlEIQgnOhBaBp2+R70v1RwrZ1iAIUm6DrQg6ZrvP5PX8EuNDGiJnpIR0mF0AUTP72P+HCS1//pK1QXykT7Yjzx9z/CgV95hLf+xWso79YEzynbXPiTswx8JM/jv/X0tu/7jw7S/+QQr/1PPyB3bo1oX4xnfvsYmvV+BkLvDLVFKkMpFWQZdbwXbo9UhbsjRHqipPd2BXGIoymUVNSzNZq5wLKkhw0iPVHMmImSiuZ6nWahNT+FILkrSWO9jhm3CKXCSNentlJtkxsrFSLaG2Pkhd2sn1ml+1Avyu9sQ7N0Ij0RrEQIJDQLdRq5RserzoyZRHpjGCEd35E083Xsst0+JjYYJ5QOXEHSffALp4nFqL6fqipwzn0Nm6DNglzlMev5bcfbNIJAfaluKnYabRGoa94FlvxNa6CG3rG50jEZ0ycJixjvusfJyc21YatQa2Dd2UdYxDnt/ICSCp5j4WtMGI8wqu+nKLMU5cZG1WfWOwcIEloXmjC44p7GEmFSWjdxkSLLAjVVoqZK6Bj0aENoQmfNX9iW3WkSYkw/gI/HOfc4tVatxyX/KpPmk+zWD5GXKx21Gg1CWIR51zlOs3Vfg1vWsjKh2qr8Sa2LJBlycoWKuvv1t2ssxqOfH+GVP5jCdz/4sW8PDbEaGNA4dNhkYo/O2G6ddFojHtcoFhRTFx3y+fcnbdTzm7hug5AZJxUfodZYQ0oPTRiEwxl2Dz+PaUY/QFaau4MvHWqNNbpSe7DMONXaCm4rs8pxqjTtAvFoP0rJIHB9i7J1tb7KWu4cowPPsmf0U1xdeJlqfaUVYyLQNIOQlSAa7iZfmn7P3IFus0ph8RwDk8cYmDxGbvYdfC9IpzdCUXzXxqkX8V2b/NxpRh/7aXrGnyI/dxrpe5jhOP37P0azmqOyNv3Agl2depHi0gUyw4fJjDwSyDugiKYG6d79BNXcPPXSyq0vdAO4zQqabhLrHsWu5toxX74bLDTZM6vs+tQET/3D51g9uczq28tUF4MXsBmzSO/vJn8hS3Uh+Ft9rUb29Apjn54glAq1F+i7hTA0kuNp7EKDwlQOFDRyDbKnVxl8buServ3e4C7mhSYYenaEwedGSY2nifREeeQ/ehzpSWa/Nc38y7NohsbkLx1i4CMj6KaG0DWcis2Z3ztJ/tI6Rljn6d8+Rv7COvHhBKFUCCNqsnZqhTP/8iS+49P/5BCjL4yR3tuFlbBI7+tCupKZb11h4eVZhC4Y/8m97Pr0BEZIR+gCt+5x8U/OsPzGIiiI9EZ59DefJDGSbPc9fz7Lmd8/2SZwvY8NMPLCGJn93Uz/5UXO//GZ+3h/tyMi4kREjGveRRw2N0JVVaSmyjeUnLgVqqqIrRrs0g+goZOTK9iqvo2MhUSYlNZDUWZbLrctmlhbCJiBSUbrpyqLlFWu45isv8gufZKU6KVE8J2Lg0MTAxNHNVvRWg4GJj4++h1KWYRFjISWYdmfobYlns7FZt1fos8cJSEyHcRKACtydpNUben1g0IoZjBwINVyEX5IrO4bPvWZEJ/9XJilJZ+FeZ/XTrjMznisrkoadUX9faoT6Lg11gsXiYa72D38PPFoP45XwzSipOIjSOVTKs8Rj20vKqtpBpnkBJYZQ9ctQlYSy4ghEAz2PEYyNhQE0/s2pevERaPhbpLxYXTNQtctMslxhNCIhLsYG/pYK3jepWkXKVbm2unjQuikE2OEQ8mg1IcZJxwK4nN6uw4SspLtNiu15bYSupQe9cY6Q71PIITGcvaddvCtQlKpr5BJTaCkz3rhUgeR9KXL/MprhENpetL7OLz3F6g1sjhOLUjXtuJEQmkct065utRBrCKhLlKJkW3jDLcUtX3fwZcutlOmUJ69ozR5JT2y028QimXo2/ccmeHDuHYNTTcwwwmWz38vkDFQkvzCu0TSAwxMHiM9dCCIq4pl0I0Qi2e/HQRRag8m61P6LmuXj2OFEwwf+SzdY4+jpE840YPv2Syd++42LZY7QXn1Cj3jRxk88AKpgX0o6ePUiyyceRHpu5SuFnjzd15l6GO76H96iLHPTHD1b6aY+dY0QhNohobvdN533/YRuoam3/s9EQJ0S0e6Eum22pEKz364Nys3hVTMvTTD4vF59vzsJCPPj3Hif/gBAH4zGLf0JGunV1l5a4n6Wg0zZvHMb3+M4efHKF4NrANGxKDnkV7e+cqb1JYq9D0xwKN/9ynm/v1V1t9dY+n4PNlTK7ywK8Xsi1eY/dY0Sqp2G8pX5C+uU54NYur0sM4T//lHGP3kONmza3h1l64DPXQd7OXkPz9O+VoJK2FhJkIdc2LupRmyZ1Z5+h999IaxcfcTujDR0HFodBAZHw9POZgifFfXrakSl713GNb3sFs/xIi+l7xcZdWfo6xy7c2ViYUmDBqyetMNlyFMDExqqrTtOA8HDw9LhNvSEIpN66dqheO3PrT+d2dhJyYWAg1HNbmesLjYSOS2zFRFEGv3oCE0sKIGmi7QzU0rH4AZ1jHDOr4ncepeW8ZPMwShqIHQBL4rseseKDBCGmZER8lACkb6CgRYER3fVZgRHSHArra+u0c8NMTqG19v8tJ3bMoVRb2mHkACk2qVWbmR5atl0ld+xwMgpcvC6huBZkvXQQZ7Hw+yHtw65erFgtGmAAARYUlEQVQi15ZfJRkbJhLp3hYUbxoRJkY+QTTSHShqC70VL6MY6jvaak/i+TaXr73YQay6UnuYGP1UoBQtWsraShEJZZgY/VT73GL5GtX6atuCpGsmY0MfJZXY1WpTQ6ChlKS/+zB9XQdRSuJLNygxs7LhYlLUm3l86aIJPcj623KvKtWlQCkYRbW+XduoaReZmv0G1b6jdKf3EYv0k4qHAIXj1Wk0C+RK09tquKWTu9k39rnWOLWWxpAibKWYGP1ke5zl6gKV2vKONeBuBqde5Nrbf01qYB/x7jEMKxKo6C+co7y2WbrIdxosnv021ew14r270c0QpeVLFBcvBNmASiKkoJqbx6kHlhslPcpr0209KOk7lJYubbMuKeVTyc7QrOZuqKvSrKxz7e2/IjU4Sbx7F0LTyF59i+LyRZqVzqBlu1Ygf+0UbvP2Mlka5VVm3/wa6eFDhGIZpO9SL612/r7zZS796btc+3aEvV84wN4vHGT5tUXcuktjvU5sIIFu6fiOj2ZqxAbjOBUbt3bvEijSVzRydcyERSgdxi400UyNaN/90IUTaEJH1y10zUQIDdOI4Lo1pPI67oEQGpowMI0wQgh03cTQg/kSuP7u70vJa3j4to/X9JCexClvd1UXpnIkx1J0Tfaghwx8xyfSE0HbQl6WXlsgd24tEGh9exnf9ghlAmLhNz2kK1Gewmt4gftOdo6jNFsMkhf2dWGEDZQnCaXD6KaGR2ChlJ5k+NguYI78pRzetU7JGun4uFUHeRtu4fsCtdXtttXKIbiXrE+FIi9XKMscCZGhTx+lRxuiWxtkyns7CL5nI9NW7aiVtRVSBQrvO7rhN+Inke3eb59h9zbnNojZThY8QVCFYLtrVD0w6/xW7DrazWM/O4qUCrfptQPa08NRnviFXST6w0hXcubrC8yfyqObGo/+zAi7jnajpKK4WOf1fzODEPDMr46THomipOLqiXXOf3uJUNzguS/vobhYp29/El3XeP3fXGX96q3Let0KDw2xKhSCifqgsF64TK3+Ry25he2xWp5nMzX7DXTN2qbMbDtlpue/w+LqWxhG64Xl29hOBV861Jt5StX5tmDmBhy3zoWr/w5Nu9XPoFqigptYy1+gXFu65bh83+4oEeP7DpevfRtdv3Xgr213vhyL5VneufBHANtq+uVKVzh5/l+hlKLR3NmXbjtlZha+z+Lqm5hGDF0L0od938H1arje9l1TrjhFrbF2G+N07roUju/Uyc+dJj+3PUuk4zjXJj9/hvz8zm4MpXzWr77Z/iw9h5WLP2h/9uwaC2df3H6e77F6+fgt++k2K6zPvMX6zFs3Pa66fi1QY79dKEUtv9ChMr8BoQtGnh/DiJhUF8topk6kN4bXcJGej9/0mP/eLI/+3SeZ/OXDZM+skt7XzcAzw1z+8wu4NRehC6K9MaxUiFA6jBExSY2n8Zoe9bUayldYyRDhTJj4UAIjbJDYlUL6KpBpqLlkT6+y9wsHOPCrR1g+MU98JEnf4wPbLGV3irGhY/R1HWy5o5MYeohH9v4ivnSw3SqXZv6Gpl0gHEozufunCVlJdD2EYUQY6H2crtQepPRZzZ1lbvnWv+H9RCgT5vHfeppoX4zqYgWnYmMlQzSynZuL+moN1SJL0pNBbJx2e+TCSlgc/rXHyUz2UF0oY5eahDKRgDC3LlG8kuet/+04uz+7h0d/80ncusulP32X1beW2+2+13Bo4uIQE6mW1EIwT0wC+YHrpQbuFB4uBbVGwcuSEb0cNp9jUJ9oEytbNdpB6AbmDeUbPBzqqkJUJDCwOmKgIiKOgUldVe6pyPXN4NDAUU1iWhLd17cElQtiIoFA2ya5cWe4zQSh6xBOmjz5S2PMvpFj+sQahz83zMBkCjOsc/SLu7DrHi9/5RIjj2b4yN+ZID9Xo38yyYFPDvLd//0CjbKD0ARO3ePpXxknkrJ45Q8uE46bvPD39rM+W6Wy1qRvbxIl4bU/uhrELxbufSMIDxGxetC4UcX7DShkO1h7x++VDEq47BD/7HkNKjvEDSnlU63fXWyM41Zw3Duf8ME4bk1UdoIvHSo3IHO+71Cu7pADu0MPHLeG496e62ojdf5DvE9QgeVk92f3YKXCKF9Sni1y6itv4pSDl9DqySXe+T99Jn5mP4PPjtBYr3P+j06zdDxIcQ6lwhz41UeIDycxExYCePy3nqGerXH2/36bZr7B0EdHGf3EbsyoiVN12PfFg3gNj8t/cYGVNxYpXS3w9r94nX1fOsjhX3+C/PksU189T/fh3k4dsC3w/CbT8y8F0gY3qHuYK061nsHtL36pvLbIrevWmVt+7YaboA1rcqF0lfNXvkajVcR5NXeWYnm2PYfnl19D160bSD/cGfqPDtL3+ACv/ncvUbiUAwHP/eMXtlsTbmneDywQQohteXDdh/sY/vgYb/3z46y8ETzfR//+R4gPJzbP9hW5d9eCpILeGOM/vY/Hf+tpXv3H36N6m1mh9xt1VaUks/TpI0HpHZlFIBjQdxMVCapquwh0YB/SWv/WOgjZBpKiGw2dJjV85bXK9gQ6hb7anGMONqv+NcaNR5gwjrDoT+MpByE0QoRxW4TKw2PJn+GA+TSj+n6W5QxSeYRElN36IWqq1JZFeBBoqBprcp5BfZxBfYKsv4BCkdS6GNL3UJCrVOT2guS3C1s10DHIaH00/VowzxAdcW87IZq2iKQsZl7PUlpqcO1kjt1PdxNOmAw/msGueqQHo5gRnXDCJJq2GDqcZu6dPKtTnXNu9LEMZ76+QP5aDc0Q5OZqDOxPUl1r4jZ9rr6epbh4f12bHxKrD/EhPsQNoaRi+Y1F1t5ehg1tGV8i3c0dv/IV2VMr5M6tIXQtyKxy/fYK3Sw0OPW7b26XGFCqbXGa+85V5l+e3db+1uusnVpm/dwaQhOoluXl2neudpSz6by8vOFGYAPV+uqOruvr4UuHQnm7JtT1sN0K9pYNT6OZ77Dg3u2mZicEcSKCUDpMbCBOZrKbrgM9rL97Z20oqWjmG/Q80sfaqRV828OpOLhVB+VLUIpQMkS0L0Z6T4bexwZo5jcXoq6DPZhRk9pKNSh+W7YB0Y6vE5rAjFuEuyLolo4ZNQlnwviu3KZVdr+gkMx45zloPs1+42gQaK0UDjYFuYYpNkWbQ0TZbRzEwCIsohhYbU0nT7nk5DJrMtgkpLVeRvX9eLgdRKlJrUOWQCFZ8q9iiTD92i56tWFcXDQ0DEymvTMtS5BiXS4y5ycY0ifo1UeQeJiEkPhc9k5RV9VbuhTv5T7Ne1MYmIzpBxnUx1EoQoRpqCrT3plbiqXeDDm5TJ8cZZdxgD59FKl86qrClPf2DQVVIZjbSiqMcJD1a1gami6QvsKuulw9sc7CmeC58j1FabmB2/SJZULbVDKcho8VNVoVNQRmRMdpbugMKjz7/runPyRWH+JDfIibQ6rbcrlJV4K7w0tKBcHsNz3Xk3Cr+BsVxOp0/OkeXYEPA3zb2zFWbf3sKqsnlzj85cfxGi71tRrzL89iRIy2kcqtuvjulnukVBDrtPV3UnDlLy9y6D94jI/818dway5Tf36epR/Ok5/KsfDKHHu/eJDxn9pHY73O4qtzRAdi7YDhaG+MfV88iB422mT58l9eoLoUWA6ifTEO/O0jxPrjQXZiJkxsMEFppsD5Pzrd2Zf7iIrKc849QZc20BYIXfeXiGlJIiK+ZWEPVMMlEkc1KfubRFi1QsQ3sOJfw1Z1oiKBLkyUlNRVhaLKdmTOQeCOnPbOsC6WSGpdGMLEVz4NVSUvN+tl+nhc8y5Q0tZJii50EQiE5uVqm3wpJDm5TEW1annis9bSnILANbnkz2xz28lWdqEl1m/oTrSpc8U7zbq2RFJkEGjUVJmiWqN5XZB6TZW45l+goW7P41BXFS54b9KlDRAWEaQKNK5u5Yqt5W0KC3WO/PQI195aZ+K5XoSAZsVl9o0c/ZNJCos1hBBIT1KYrzH3dp6f+K1JHvnJYWp5G00XzJ8qMPX9VQ58agC36ROKG8QyFsvn765k1e1C3MiM/l5CCPH+d+JDfIgP8SE+gLBSIay4RXVxu+t/wxIkBDQLTdT/397dhWZZxnEc//62Z9PNmdoqX6ZlkSQWZGFhFBLagZVkB9ELRRJFJ0EvFGGdddBBEFlRCKGFRWRhQhEURHmQUKImvZo0LHWiLts00znd9u/gvqaPbwfq0+7nfvx9YOy5rus+uMaf/73/ruu67/UPUGpuoGd39guxpe08evccPLIypDrRMmEkB7t7OLz/6EqE6kTThc00NDcw0D9Az+4e+g5k46WmEsNbm6kr1dG75yD9vX00jhxGz+4DxEBk77lqbaI0PPs7ffCBhsGVxPph9YwYP/KE///Y39ufvbbDd387idbJLVw5dwINw+rpbN9H85gG1i7/k8amElNmXcS4qaPo7wu2behi87edxABMvq41K8LqRNeW/fz8+Xb6Dg8wdfY4xk8bTf+hAdpX76Ljh24aR5SYPn8Sv6/upGvLGT1VvT4iZpxswIWVmZmZ2ek5ZWH1/79QxMzMzOwc4cLKzMzMrEKq5fD6v8CmvCdhZ+wC4Cz+WZ3lzPErLseu2By/4rrkVAPVUlhtOtVepVU/Sescv+Jy/IrLsSs2x682eSvQzMzMrEJcWJmZmZlVSLUUVm/lPQE7K45fsTl+xeXYFZvjV4Oq4j1WZmZmZrWgWlaszMzMzAov98JK0lxJmyS1S1qY93zsWJImSVol6VdJv0h6IvWfL+lLSb+n72NSvyS9nuL5o6Rr8/0JDEBSvaQNkj5L7UslrUlx+lBSY+ofltrtaXxynvM+10kaLWmFpN8kbZR0g3OvOCQ9le6bP0v6QNJw517ty7WwklQPvAncCkwD7pM0Lc852Qn6gKcjYhowE3gsxWgh8FVETAG+Sm3IYjklfT0KLB76KdtJPAFsLGu/BCyKiMuBbuDh1P8w0J36F6XrLD+vAV9ExFTgarIYOvcKQFIb8DgwIyKuAuqBe3Hu1by8V6yuB9ojYnNEHAKWA/NznpOViYgdEfF9+ryP7MbeRhanZemyZcCd6fN84N3IfAeMljR+iKdtZSRNBG4HlqS2gNnAinTJ8fEbjOsKYE663oaYpFHALGApQEQciog9OPeKpAQ0SSoBzcAOnHs1L+/Cqg3YVtbuSH1WhdLS9DXAGmBsROxIQzuBsemzY1p9XgWeBQZSuxXYExF9qV0eoyPxS+N70/U29C4F/gLeSdu4SySNwLlXCBGxHXgZ2EpWUO0F1uPcq3l5F1ZWEJJagI+BJyPin/KxyB4t9eOlVUjSPKAzItbnPRc7bSXgWmBxRFwD7Ofoth/g3Ktm6ezbfLICeQIwApib66RsSORdWG0HJpW1J6Y+qyKSGsiKqvcjYmXq3jW4zZC+d6Z+x7S63AjcIelPsq322WTndkan7Qk4NkZH4pfGRwF/D+WE7YgOoCMi1qT2CrJCy7lXDLcAf0TEXxFxGFhJlo/OvRqXd2G1FpiSnpJoJDvY92nOc7IyaY9/KbAxIl4pG/oUWJA+LwA+Ket/MD2hNBPYW7ZtYUMsIp6LiIkRMZksv76OiPuBVcBd6bLj4zcY17vS9V4RyUFE7AS2Sboidc0BfsW5VxRbgZmSmtN9dDB+zr0al/sLQiXdRnYGpB54OyJezHVCdgxJNwHfAD9x9IzO82TnrD4CLga2AHdHRFe6gbxBtuR9AHgoItYN+cTtBJJuBp6JiHmSLiNbwTof2AA8EBG9koYD75GdpesC7o2IzXnN+VwnaTrZQweNwGbgIbI/iJ17BSDpBeAesqerNwCPkJ2lcu7VsNwLKzMzM7NakfdWoJmZmVnNcGFlZmZmViEurMzMzMwqxIWVmZmZWYW4sDIzMzOrEBdWZmZmZhXiwsrMzMysQlxYmZmZmVXIf3iIzhLOea1lAAAAAElFTkSuQmCC\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "source": [
        "#word cloud for positive review words\n",
        "plt.figure(figsize=(10,10))\n",
        "positive_text=norm_train_reviews[1]\n",
        "WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)\n",
        "positive_words=WC.generate(positive_text)\n",
        "plt.imshow(positive_words,interpolation='bilinear')\n",
        "plt.show"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "j4nUyTDqyM9s"
      },
      "source": [
        "**Word cloud for negative review words**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 75,
      "metadata": {
        "trusted": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 348
        },
        "id": "hXt5YwxryM9s",
        "outputId": "ce603fd5-6893-4d68-9455-78f4148f5ca4"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<function matplotlib.pyplot.show(*args, **kw)>"
            ]
          },
          "metadata": {},
          "execution_count": 75
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x720 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "source": [
        "#Word cloud for negative review words\n",
        "plt.figure(figsize=(10,10))\n",
        "negative_text=norm_train_reviews[8]\n",
        "WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)\n",
        "negative_words=WC.generate(negative_text)\n",
        "plt.imshow(negative_words,interpolation='bilinear')\n",
        "plt.show"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "O9rGfYaxyM9s"
      },
      "source": [
        "**Conclusion:**\n",
        "* We can observed that both logistic regression and multinomial naive bayes model performing well compared to linear support vector  machines.\n",
        "* Still we can improve the accuracy of the models by preprocessing data and by using lexicon models like Textblob."
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3.9.12 ('base')",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.9.12 (main, Apr  4 2022, 05:22:27) [MSC v.1916 64 bit (AMD64)]"
    },
    "vscode": {
      "interpreter": {
        "hash": "5a8404f3db4fecb075c6fddddb906e868a9e2f134ee8a044c603f590c8800039"
      }
    },
    "colab": {
      "provenance": [],
      "include_colab_link": true
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}