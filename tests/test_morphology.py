from toygger.morphology import Tagset


def test_Tagset():
    class TestTagset(Tagset):

        def __init__(self):
            number = ('sg', 'pl', 'NOVAL')
            gender = ('m1', 'm2', 'm3', 'f', 'NOVAL')
            another_category = ('x', 'y', 'NOVAL')
            yet_another_category = ('k', 'l', 'NOVAL')

            optional = Tagset._optional

            combinations = {
                'adj': [number,
                        optional(another_category),
                        optional(yet_another_category)],
                'subst': [number, gender]
            }
            super().__init__(lexemes={}, combinations=combinations)

    x = TestTagset()

    assert x.valid_ctags == set({
        'adj:pl',
        'adj:pl:k',
        'adj:pl:l',
        'adj:pl:x',
        'adj:pl:x:k',
        'adj:pl:x:l',
        'adj:pl:y',
        'adj:pl:y:k',
        'adj:pl:y:l',
        'adj:sg',
        'adj:sg:k',
        'adj:sg:l',
        'adj:sg:x',
        'adj:sg:x:k',
        'adj:sg:x:l',
        'adj:sg:y',
        'adj:sg:y:k',
        'adj:sg:y:l',
        'subst:pl:f',
        'subst:pl:m1',
        'subst:pl:m2',
        'subst:pl:m3',
        'subst:sg:f',
        'subst:sg:m1',
        'subst:sg:m2',
        'subst:sg:m3',
    })
