# Citing pop-cosmos
If you make use of pop-cosmos, please cite the papers describing the model, and the relevant dependencies. Below is a string of LaTeX code that could be used in, e.g., a software acknowledgements section.
```latex
\texttt{astropy} \citep{astropy13, astropy18, astropy22};
\texttt{matplotlib} \citep{hunter07};
\texttt{numpy} \citep{harris20};
\texttt{pop-cosmos} \citep{alsing24, thorp24, thorp25, deger25};
\texttt{scipy} \citep{virtanen20};
\texttt{speculator} \citep{alsing20};
\texttt{torch} \citep{paszke19};
\texttt{torchdiffeq} \citep{chen18}.
```
We would also encourage users to acknowledge `fsps`, `prospector`, and `sedpy`, upon which our SPS model is based.
```latex
\texttt{fsps} \citep{conroy09, conroy10a, conroy10b};
\texttt{prospector} \citep{johnson21a};
\texttt{python-fsps} \citep{johnson21b};
\texttt{sedpy} \citep{johnson21c}.
```
BibTeX entries for all of these references are included below, based on NASA ADS.
```bibtex
@ARTICLE{alsing20,
       author = {{Alsing}, Justin and {Peiris}, Hiranya and {Leja}, Joel and {Hahn}, ChangHoon and {Tojeiro}, Rita and {Mortlock}, Daniel and {Leistedt}, Boris and {Johnson}, Benjamin D. and {Conroy}, Charlie},
        title = "{SPECULATOR: Emulating Stellar Population Synthesis for Fast and Accurate Galaxy Spectra and Photometry}",
      journal = {\apjs},
     keywords = {Galaxies, Neural networks, Galaxy photometry, 573, 1933, 611, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2020,
        month = jul,
       volume = {249},
       number = {1},
          eid = {5},
        pages = {5},
          doi = {10.3847/1538-4365/ab917f},
archivePrefix = {arXiv},
       eprint = {1911.11778},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020ApJS..249....5A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{alsing24,
       author = {{Alsing}, Justin and {Thorp}, Stephen and {Deger}, Sinan and {Peiris}, Hiranya V. and {Leistedt}, Boris and {Mortlock}, Daniel and {Leja}, Joel},
        title = "{pop-cosmos: A Comprehensive Picture of the Galaxy Population from COSMOS Data}",
      journal = {\apjs},
     keywords = {Galaxy evolution, Galaxy abundances, Galaxy chemical evolution, Cosmological parameters, Cosmology, Redshift surveys, 594, 574, 580, 339, 343, 1378, Astrophysics - Astrophysics of Galaxies, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2024,
        month = sep,
       volume = {274},
       number = {1},
          eid = {12},
        pages = {12},
          doi = {10.3847/1538-4365/ad5c69},
archivePrefix = {arXiv},
       eprint = {2402.00935},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJS..274...12A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{astropy13,
       author = {{Astropy Collaboration} and {Robitaille}, Thomas P. and {Tollerud}, Erik J. and {Greenfield}, Perry and {Droettboom}, Michael and {Bray}, Erik and {Aldcroft}, Tom and {Davis}, Matt and {Ginsburg}, Adam and {Price-Whelan}, Adrian M. and {Kerzendorf}, Wolfgang E. and {Conley}, Alexander and {Crighton}, Neil and {Barbary}, Kyle and {Muna}, Demitri and {Ferguson}, Henry and {Grollier}, Fr{\'e}d{\'e}ric and {Parikh}, Madhura M. and {Nair}, Prasanth H. and {Unther}, Hans M. and {Deil}, Christoph and {Woillez}, Julien and {Conseil}, Simon and {Kramer}, Roban and {Turner}, James E.~H. and {Singer}, Leo and {Fox}, Ryan and {Weaver}, Benjamin A. and {Zabalza}, Victor and {Edwards}, Zachary I. and {Azalee Bostroem}, K. and {Burke}, D.~J. and {Casey}, Andrew R. and {Crawford}, Steven M. and {Dencheva}, Nadia and {Ely}, Justin and {Jenness}, Tim and {Labrie}, Kathleen and {Lim}, Pey Lian and {Pierfederici}, Francesco and {Pontzen}, Andrew and {Ptak}, Andy and {Refsdal}, Brian and {Servillat}, Mathieu and {Streicher}, Ole},
        title = "{Astropy: A community Python package for astronomy}",
      journal = {\aap},
     keywords = {methods: data analysis, methods: miscellaneous, virtual observatory tools, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2013,
        month = oct,
       volume = {558},
          eid = {A33},
        pages = {A33},
          doi = {10.1051/0004-6361/201322068},
archivePrefix = {arXiv},
       eprint = {1307.6212},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2013A&A...558A..33A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{astropy18,
       author = {{Astropy Collaboration} and {Price-Whelan}, A.~M. and
         {Sip{\H{o}}cz}, B.~M. and {G{\"u}nther}, H.~M. and {Lim}, P.~L. and
         {Crawford}, S.~M. and {Conseil}, S. and {Shupe}, D.~L. and
         {Craig}, M.~W. and {Dencheva}, N. and {Ginsburg}, A. and {Vand
        erPlas}, J.~T. and {Bradley}, L.~D. and {P{\'e}rez-Su{\'a}rez}, D. and
         {de Val-Borro}, M. and {Aldcroft}, T.~L. and {Cruz}, K.~L. and
         {Robitaille}, T.~P. and {Tollerud}, E.~J. and {Ardelean}, C. and
         {Babej}, T. and {Bach}, Y.~P. and {Bachetti}, M. and {Bakanov}, A.~V. and
         {Bamford}, S.~P. and {Barentsen}, G. and {Barmby}, P. and
         {Baumbach}, A. and {Berry}, K.~L. and {Biscani}, F. and {Boquien}, M. and
         {Bostroem}, K.~A. and {Bouma}, L.~G. and {Brammer}, G.~B. and
         {Bray}, E.~M. and {Breytenbach}, H. and {Buddelmeijer}, H. and
         {Burke}, D.~J. and {Calderone}, G. and {Cano Rodr{\'\i}guez}, J.~L. and
         {Cara}, M. and {Cardoso}, J.~V.~M. and {Cheedella}, S. and {Copin}, Y. and
         {Corrales}, L. and {Crichton}, D. and {D'Avella}, D. and {Deil}, C. and
         {Depagne}, {\'E}. and {Dietrich}, J.~P. and {Donath}, A. and
         {Droettboom}, M. and {Earl}, N. and {Erben}, T. and {Fabbro}, S. and
         {Ferreira}, L.~A. and {Finethy}, T. and {Fox}, R.~T. and
         {Garrison}, L.~H. and {Gibbons}, S.~L.~J. and {Goldstein}, D.~A. and
         {Gommers}, R. and {Greco}, J.~P. and {Greenfield}, P. and
         {Groener}, A.~M. and {Grollier}, F. and {Hagen}, A. and {Hirst}, P. and
         {Homeier}, D. and {Horton}, A.~J. and {Hosseinzadeh}, G. and {Hu}, L. and
         {Hunkeler}, J.~S. and {Ivezi{\'c}}, {\v{Z}}. and {Jain}, A. and
         {Jenness}, T. and {Kanarek}, G. and {Kendrew}, S. and {Kern}, N.~S. and
         {Kerzendorf}, W.~E. and {Khvalko}, A. and {King}, J. and {Kirkby}, D. and
         {Kulkarni}, A.~M. and {Kumar}, A. and {Lee}, A. and {Lenz}, D. and
         {Littlefair}, S.~P. and {Ma}, Z. and {Macleod}, D.~M. and
         {Mastropietro}, M. and {McCully}, C. and {Montagnac}, S. and
         {Morris}, B.~M. and {Mueller}, M. and {Mumford}, S.~J. and {Muna}, D. and
         {Murphy}, N.~A. and {Nelson}, S. and {Nguyen}, G.~H. and
         {Ninan}, J.~P. and {N{\"o}the}, M. and {Ogaz}, S. and {Oh}, S. and
         {Parejko}, J.~K. and {Parley}, N. and {Pascual}, S. and {Patil}, R. and
         {Patil}, A.~A. and {Plunkett}, A.~L. and {Prochaska}, J.~X. and
         {Rastogi}, T. and {Reddy Janga}, V. and {Sabater}, J. and
         {Sakurikar}, P. and {Seifert}, M. and {Sherbert}, L.~E. and
         {Sherwood-Taylor}, H. and {Shih}, A.~Y. and {Sick}, J. and
         {Silbiger}, M.~T. and {Singanamalla}, S. and {Singer}, L.~P. and
         {Sladen}, P.~H. and {Sooley}, K.~A. and {Sornarajah}, S. and
         {Streicher}, O. and {Teuben}, P. and {Thomas}, S.~W. and
         {Tremblay}, G.~R. and {Turner}, J.~E.~H. and {Terr{\'o}n}, V. and
         {van Kerkwijk}, M.~H. and {de la Vega}, A. and {Watkins}, L.~L. and
         {Weaver}, B.~A. and {Whitmore}, J.~B. and {Woillez}, J. and
         {Zabalza}, V. and {Astropy Contributors}},
        title = "{The Astropy Project: Building an Open-science Project and Status of the v2.0 Core Package}",
      journal = {\aj},
     keywords = {methods: data analysis, methods: miscellaneous, methods: statistical, reference systems, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2018,
        month = sep,
       volume = {156},
       number = {3},
          eid = {123},
        pages = {123},
          doi = {10.3847/1538-3881/aabc4f},
archivePrefix = {arXiv},
       eprint = {1801.02634},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2018AJ....156..123A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{astropy22,
       author = {{Astropy Collaboration} and {Price-Whelan}, Adrian M. and {Lim}, Pey Lian and {Earl}, Nicholas and {Starkman}, Nathaniel and {Bradley}, Larry and {Shupe}, David L. and {Patil}, Aarya A. and {Corrales}, Lia and {Brasseur}, C.~E. and {N{"o}the}, Maximilian and {Donath}, Axel and {Tollerud}, Erik and {Morris}, Brett M. and {Ginsburg}, Adam and {Vaher}, Eero and {Weaver}, Benjamin A. and {Tocknell}, James and {Jamieson}, William and {van Kerkwijk}, Marten H. and {Robitaille}, Thomas P. and {Merry}, Bruce and {Bachetti}, Matteo and {G{"u}nther}, H. Moritz and {Aldcroft}, Thomas L. and {Alvarado-Montes}, Jaime A. and {Archibald}, Anne M. and {B{'o}di}, Attila and {Bapat}, Shreyas and {Barentsen}, Geert and {Baz{'a}n}, Juanjo and {Biswas}, Manish and {Boquien}, M{'e}d{'e}ric and {Burke}, D.~J. and {Cara}, Daria and {Cara}, Mihai and {Conroy}, Kyle E. and {Conseil}, Simon and {Craig}, Matthew W. and {Cross}, Robert M. and {Cruz}, Kelle L. and {D'Eugenio}, Francesco and {Dencheva}, Nadia and {Devillepoix}, Hadrien A.~R. and {Dietrich}, J{"o}rg P. and {Eigenbrot}, Arthur Davis and {Erben}, Thomas and {Ferreira}, Leonardo and {Foreman-Mackey}, Daniel and {Fox}, Ryan and {Freij}, Nabil and {Garg}, Suyog and {Geda}, Robel and {Glattly}, Lauren and {Gondhalekar}, Yash and {Gordon}, Karl D. and {Grant}, David and {Greenfield}, Perry and {Groener}, Austen M. and {Guest}, Steve and {Gurovich}, Sebastian and {Handberg}, Rasmus and {Hart}, Akeem and {Hatfield-Dodds}, Zac and {Homeier}, Derek and {Hosseinzadeh}, Griffin and {Jenness}, Tim and {Jones}, Craig K. and {Joseph}, Prajwel and {Kalmbach}, J. Bryce and {Karamehmetoglu}, Emir and {Ka{l}uszy{'n}ski}, Miko{l}aj and {Kelley}, Michael S.~P. and {Kern}, Nicholas and {Kerzendorf}, Wolfgang E. and {Koch}, Eric W. and {Kulumani}, Shankar and {Lee}, Antony and {Ly}, Chun and {Ma}, Zhiyuan and {MacBride}, Conor and {Maljaars}, Jakob M. and {Muna}, Demitri and {Murphy}, N.~A. and {Norman}, Henrik and {O'Steen}, Richard and {Oman}, Kyle A. and {Pacifici}, Camilla and {Pascual}, Sergio and {Pascual-Granado}, J. and {Patil}, Rohit R. and {Perren}, Gabriel I. and {Pickering}, Timothy E. and {Rastogi}, Tanuj and {Roulston}, Benjamin R. and {Ryan}, Daniel F. and {Rykoff}, Eli S. and {Sabater}, Jose and {Sakurikar}, Parikshit and {Salgado}, Jes{'u}s and {Sanghi}, Aniket and {Saunders}, Nicholas and {Savchenko}, Volodymyr and {Schwardt}, Ludwig and {Seifert-Eckert}, Michael and {Shih}, Albert Y. and {Jain}, Anany Shrey and {Shukla}, Gyanendra and {Sick}, Jonathan and {Simpson}, Chris and {Singanamalla}, Sudheesh and {Singer}, Leo P. and {Singhal}, Jaladh and {Sinha}, Manodeep and {Sip{H{o}}cz}, Brigitta M. and {Spitler}, Lee R. and {Stansby}, David and {Streicher}, Ole and {{\v{S}}umak}, Jani and {Swinbank}, John D. and {Taranu}, Dan S. and {Tewary}, Nikita and {Tremblay}, Grant R. and {Val-Borro}, Miguel de and {Van Kooten}, Samuel J. and {Vasovi{'c}}, Zlatan and {Verma}, Shresth and {de Miranda Cardoso}, Jos{'e} Vin{'i}cius and {Williams}, Peter K.~G. and {Wilson}, Tom J. and {Winkel}, Benjamin and {Wood-Vasey}, W.~M. and {Xue}, Rui and {Yoachim}, Peter and {Zhang}, Chen and {Zonca}, Andrea and {Astropy Project Contributors}},
        title = "{The Astropy Project: Sustaining and Growing a Community-oriented Open-source Project and the Latest Major Release (v5.0) of the Core Package}",
      journal = {\apj},
     keywords = {Astronomy software, Open source software, Astronomy data analysis, 1855, 1866, 1858, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2022,
        month = aug,
       volume = {935},
       number = {2},
          eid = {167},
        pages = {167},
          doi = {10.3847/1538-4357/ac7c74},
archivePrefix = {arXiv},
       eprint = {2206.14220},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022ApJ...935..167A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@INPROCEEDINGS{chen18,
       author = {Chen, Ricky T. Q. and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David K},
    booktitle = {Advances in Neural Information Processing Systems},
       editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
        pages = {6572--6583},
    publisher = {Curran Associates, Inc.},
        title = {Neural Ordinary Differential Equations},
       volume = {31},
         year = {2018},
archivePrefix = {arXiv},
       eprint = {1806.07366},
          url = {https://proceedings.neurips.cc/paper_files/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf}
}

@ARTICLE{conroy09,
       author = {{Conroy}, Charlie and {Gunn}, James E. and {White}, Martin},
        title = "{The Propagation of Uncertainties in Stellar Population Synthesis Modeling. I. The Relevance of Uncertain Aspects of Stellar Evolution and the Initial Mass Function to the Derived Physical Properties of Galaxies}",
      journal = {\apj},
     keywords = {galaxies: evolution, galaxies: stellar content, stars: evolution, Astrophysics},
         year = 2009,
        month = jul,
       volume = {699},
       number = {1},
        pages = {486-506},
          doi = {10.1088/0004-637X/699/1/486},
archivePrefix = {arXiv},
       eprint = {0809.4261},
 primaryClass = {astro-ph},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2009ApJ...699..486C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{conroy10a,
       author = {{Conroy}, Charlie and {White}, Martin and {Gunn}, James E.},
        title = "{The Propagation of Uncertainties in Stellar Population Synthesis Modeling. II. The Challenge of Comparing Galaxy Evolution Models to Observations}",
      journal = {\apj},
     keywords = {galaxies: evolution, galaxies: stellar content, Astrophysics - Cosmology and Extragalactic Astrophysics, Astrophysics - Galaxy Astrophysics},
         year = 2010,
        month = jan,
       volume = {708},
       number = {1},
        pages = {58-70},
          doi = {10.1088/0004-637X/708/1/58},
archivePrefix = {arXiv},
       eprint = {0904.0002},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2010ApJ...708...58C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{conroy10b,
       author = {{Conroy}, Charlie and {Gunn}, James E.},
        title = "{The Propagation of Uncertainties in Stellar Population Synthesis Modeling. III. Model Calibration, Comparison, and Evaluation}",
      journal = {\apj},
     keywords = {galaxies: evolution, galaxies: stellar content, stars: evolution, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2010,
        month = apr,
       volume = {712},
       number = {2},
        pages = {833-857},
          doi = {10.1088/0004-637X/712/2/833},
archivePrefix = {arXiv},
       eprint = {0911.3151},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2010ApJ...712..833C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{deger25,
       author = {{Deger}, Sinan and {Peiris}, Hiranya V. and {Thorp}, Stephen and {Mortlock}, Daniel J. and {Jagwani}, Gurjeet and {Alsing}, Justin and {Leistedt}, Boris and {Leja}, Joel},
        title = "{pop-cosmos: Star formation over 12 Gyr from generative modelling of a deep infrared-selected galaxy catalogue}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics of Galaxies, Cosmology and Nongalactic Astrophysics},
         year = 2025,
        month = sep,
          eid = {arXiv:2509.20430},
        pages = {arXiv:2509.20430},
archivePrefix = {arXiv},
       eprint = {2509.20430},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250920430D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{harris20,
       author = {{Harris}, Charles R. and {Millman}, K. Jarrod and {van der Walt}, St{\'e}fan J. and {Gommers}, Ralf and {Virtanen}, Pauli and {Cournapeau}, David and {Wieser}, Eric and {Taylor}, Julian and {Berg}, Sebastian and {Smith}, Nathaniel J. and {Kern}, Robert and {Picus}, Matti and {Hoyer}, Stephan and {van Kerkwijk}, Marten H. and {Brett}, Matthew and {Haldane}, Allan and {del R{\'\i}o}, Jaime Fern{\'a}ndez and {Wiebe}, Mark and {Peterson}, Pearu and {G{\'e}rard-Marchant}, Pierre and {Sheppard}, Kevin and {Reddy}, Tyler and {Weckesser}, Warren and {Abbasi}, Hameer and {Gohlke}, Christoph and {Oliphant}, Travis E.},
        title = "{Array programming with NumPy}",
      journal = {\nat},
     keywords = {Computer Science - Mathematical Software, Statistics - Computation},
         year = 2020,
        month = sep,
       volume = {585},
       number = {7825},
        pages = {357-362},
          doi = {10.1038/s41586-020-2649-2},
archivePrefix = {arXiv},
       eprint = {2006.10256},
 primaryClass = {cs.MS},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020Natur.585..357H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{hunter07,
       author = {{Hunter}, John D.},
        title = "{Matplotlib: A 2D Graphics Environment}",
      journal = {Computing in Science and Engineering},
     keywords = {Python, Scripting languages, Application development, Scientific programming},
         year = 2007,
        month = may,
       volume = {9},
       number = {3},
        pages = {90-95},
          doi = {10.1109/MCSE.2007.55},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2007CSE.....9...90H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{johnson21a,
       author = {{Johnson}, Benjamin D. and {Leja}, Joel and {Conroy}, Charlie and {Speagle}, Joshua S.},
        title = "{Stellar Population Inference with Prospector}",
      journal = {\apjs},
     keywords = {Galaxy evolution, Spectral energy distribution, Astronomy data modeling, 594, 2129, 1859, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2021,
        month = jun,
       volume = {254},
       number = {2},
          eid = {22},
        pages = {22},
          doi = {10.3847/1538-4365/abef67},
archivePrefix = {arXiv},
       eprint = {2012.01426},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021ApJS..254...22J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@MISC{johnson21b,
       author = {{Johnson}, B.~D. and {Foreman-Mackey}, Dan and {Sick}, Jonathan and {Leja}, Joel and {Byler}, Nell and {Walmsley}, Mike and {Tollerud}, Erik and {Leung}, Henry and {Scott}, Spencer},
        title = "{dfm/python-fsps: python-fsps}",
         year = 2021,
        month = may,
          eid = {10.5281/zenodo.4737461},
          doi = {10.5281/zenodo.4737461},
      version = {v0.4.1rc1},
 howpublished = {Zenodo},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021zndo...4737461J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@MISC{johnson21c,
       author = {{Johnson}, Benjamin D.},
        title = "{bd-j/sedpy: sedpy}",
         year = 2021,
        month = mar,
          eid = {10.5281/zenodo.4582723},
          doi = {10.5281/zenodo.4582723},
      version = {v0.2.0},
 howpublished = {Zenodo},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021zndo...4582723J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@INPROCEEDINGS{paszke19,
       author = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
    booktitle = {Advances in Neural Information Processing Systems},
       editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
        pages = {8024--8035},
    publisher = {Curran Associates, Inc.},
        title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
       volume = {32},
         year = {2019},
archivePrefix = {arXiv},
       eprint = {1912.01703},
          url = {https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf}
}

@ARTICLE{thorp24,
       author = {{Thorp}, Stephen and {Alsing}, Justin and {Peiris}, Hiranya V. and {Deger}, Sinan and {Mortlock}, Daniel J. and {Leistedt}, Boris and {Leja}, Joel and {Loureiro}, Arthur},
        title = "{pop-cosmos: Scaleable Inference of Galaxy Properties and Redshifts with a Data-driven Population Model}",
      journal = {\apj},
     keywords = {Astrostatistics techniques, Redshift surveys, Galaxy photometry, Bayesian statistics, Affine invariant, Spectral energy distribution, 1886, 1378, 611, 1900, 1890, 2129, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2024,
        month = nov,
       volume = {975},
       number = {1},
          eid = {145},
        pages = {145},
          doi = {10.3847/1538-4357/ad7736},
archivePrefix = {arXiv},
       eprint = {2406.19437},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJ...975..145T},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{thorp25,
       author = {{Thorp}, Stephen and {Peiris}, Hiranya V. and {Jagwani}, Gurjeet and {Deger}, Sinan and {Alsing}, Justin and {Leistedt}, Boris and {Mortlock}, Daniel J. and {Halder}, Anik and {Leja}, Joel},
        title = "{pop-cosmos: Insights from Generative Modeling of a Deep, Infrared-selected Galaxy Population}",
      journal = {\apj},
     keywords = {Galaxy evolution, Galaxy photometry, Redshift surveys, Astronomy data modeling, Astrostatistics, Spectral energy distribution, 594, 611, 1378, 1859, 1882, 2129},
         year = 2025,
        month = nov,
       volume = {993},
       number = {2},
          eid = {240},
        pages = {240},
          doi = {10.3847/1538-4357/ae0936},
archivePrefix = {arXiv},
       eprint = {2506.12122},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025ApJ...993..240T},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{virtanen20,
       author = {{Virtanen}, Pauli and {Gommers}, Ralf and {Oliphant}, Travis E. and {Haberland}, Matt and {Reddy}, Tyler and {Cournapeau}, David and {Burovski}, Evgeni and {Peterson}, Pearu and {Weckesser}, Warren and {Bright}, Jonathan and {van der Walt}, St{\'e}fan J. and {Brett}, Matthew and {Wilson}, Joshua and {Millman}, K. Jarrod and {Mayorov}, Nikolay and {Nelson}, Andrew R.~J. and {Jones}, Eric and {Kern}, Robert and {Larson}, Eric and {Carey}, C.~J. and {Polat}, {\.I}lhan and {Feng}, Yu and {Moore}, Eric W. and {VanderPlas}, Jake and {Laxalde}, Denis and {Perktold}, Josef and {Cimrman}, Robert and {Henriksen}, Ian and {Quintero}, E.~A. and {Harris}, Charles R. and {Archibald}, Anne M. and {Ribeiro}, Ant{\^o}nio H. and {Pedregosa}, Fabian and {van Mulbregt}, Paul and {SciPy 1. 0 Contributors}},
        title = "{SciPy 1.0: fundamental algorithms for scientific computing in Python}",
      journal = {Nature Methods},
     keywords = {Computer Science - Mathematical Software, Computer Science - Data Structures and Algorithms, Computer Science - Software Engineering, Physics - Computational Physics},
         year = 2020,
        month = feb,
       volume = {17},
        pages = {261-272},
          doi = {10.1038/s41592-019-0686-2},
archivePrefix = {arXiv},
       eprint = {1907.10121},
 primaryClass = {cs.MS},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020NatMe..17..261V},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
