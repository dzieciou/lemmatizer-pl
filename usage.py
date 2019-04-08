import tagging

tagger = tagging.Library()

for r in tagger.tag('2'):
    print(r)



