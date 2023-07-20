from bokeh.sampledata.penguins import data as df
df.head()

import param
import panel as pn

class C(param.Parameterized):
    _countries = {'Africa': ['Ghana', 'Togo', 'South Africa'],
                  'Asia'  : ['China', 'Thailand', 'Japan', 'Singapore'],
                  'Europe': ['Austria', 'Bulgaria', 'Greece', 'Switzerland']}
    
    continent = param.Selector(list(_countries.keys()), default='Asia')
    country = param.Selector(_countries['Asia'])
    
    @param.depends('continent', watch=True)
    def _update_countries(self):
        countries = self._countries[self.continent]
        self.param['country'].objects = countries
        if self.country not in countries:
            self.country = countries[0]

c = C()

pn.Row(c).show()

c.country

c.country, c.param.country.objects
