import music21 as m21


if __name__ == '__main__':
    score = m21.corpus.parse('bach/bwv65.2.xml')
    print(score.analyze('key'))
    score.show()

