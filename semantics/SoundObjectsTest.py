from SoundObjects import *
import music21 as m21

def main():
    # equivalent
    poslist = list(range(-2, 13))
    stream = m21.converter.parse("tinyNotation: 4/4 c4 d e f g a b c' d' e' f' g' a' b' c''")
    stream.show()

    clef_g = Clef([0,0,0,0], 'gClef')
    


    pass

def getNote_Test():



    pass




if __name__ == "__main__":
    main()