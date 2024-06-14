from pykospacing import Spacing
spacing = Spacing()

blue_archive = """아즈사쨩이살인자가되는건싫어요...
그런어둡고우울한이야기,전싫어요
그게진실이라고,이세계의본질이라고해도,전싫어요!
저는좋아하는게있어요!
평범하고,별다른개성도없는저이지만....자신이좋아하는것에한해선절대로양보못해요!
우정으로고난을극복하여
노력이제대로보답받아서
힘든일은위로받고,친구들과서로위로해주고......!
괴로운일이있어도....누구나가최후엔,웃는얼굴이될수있는!
그런해피엔딩을전좋아해요!!
누가뭐라고말해도,몇번이고말해보이겠어요!
저희가그려나가는이야기는,저희가정하는거에요!
끝내거나하지않을거에요,앞으로도계속이어나갈거에요!
우리들의이야기......
우리들의,청춘의이야기를!!"""

hifumi_daisuki = ''

for hifumi_word in blue_archive.split('\n'):
    hifumi_daisuki += spacing(hifumi_word) + '\n'
    
print(hifumi_daisuki)