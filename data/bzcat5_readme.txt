VII/274             The Roma BZCAT - 5th edition                (Massaro+, 2015)
================================================================================
The 5th edition of the Roma-BZCAT.
     Massaro E., Maselli A., Leto C., Marchegiani P., Perri M., Giommi P.,
     Piranomonte S.
     <Astrophys. Space Sci., 357, 75 (2015)>
     =2015Ap&SS.357...75M
     =2016yCat.7274....0M
================================================================================
ADC_Keywords: QSOs ; BL Lac objects ; Galaxy catalogs ; Redshifts ; Photometry
Keywords: BL Lacertae objects: general - galaxies: quasars: general - catalogs

Abstract:
    The Roma-BZCAT is now at the 5th Edition which contains coordinates
    and multi-frequency data of 3561 sources, about 30% more than in the
    1st edition, either confirmed blazars or exhibiting characteristics
    close to this type of sources. With respect to the previous editions,
    this new edition has relevant changes in the sources' classification
    and has a new format for the notes in the tables. We emphasize that
    all the sources in the Roma-BZCAT have a detection in the radio
    band. Moreover, a complete spectroscopic information is published
    and could be accessed by us for all of them, with the exception of BL
    Lac candidates. Consequently, peculiar sources as the so called
    "radio quiet BL Lacs", which are reported in some other
    catalogues, are not included here because of possible contamination
    with hot stars and other extragalactic objects.

Description:
    In the 5th Edition we use similar denomination of
    blazars adopted in the previous editions. Each blazar
    is identified by a code, with 5BZ for all blazars, a fourth
    letter that specifies the type (B, G, Q or U), followed by the
    truncated equatorial coordinates (J2000). We introduced the edition
    number before the letters BZ to avoid possible confusion due to the
    fact that several sources changed their old names because of the new
    adopted classification.

    The codes are defined in the "Note (G1)" below.

    The 5th edition contains 1151 BZB sources, 92 of which are reported as
    candidates because we could not find their optical spectra in the
    literature, 1909 BZQ sources, 274 BZG sources and 227 BZU objects

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
bzcat5.dat       133     3561   The Roma BZCAT - 5th edition
--------------------------------------------------------------------------------

See also:
 J/A+A/495/691 : Multifrequency catalogue of blazars, Roma-BZCAT (Massaro+ 2009)
 http://www.asdc.asi.it/bzcat : BZCAT Home Page

Byte-by-byte Description of file: bzcat5.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1-  4  I4    ---     Seq       Sequential number
   6- 20  A15   ---     Name      Source name (based on J2000 position),
                                   5BZA JHHMM+DDMM (G1)
  22- 23  I2    h       RAh       Right ascension (J2000.0)
  25- 26  I2    min     RAm       Right ascension (J2000.0)
  28- 32  F5.2  s       RAs       Right ascension (J2000.0)
      34  A1    ---     DE-       Declination sign (J2000.0)
  35- 36  I2    deg     DEd       Declination (J2000.0)
  38- 39  I2    arcmin  DEm       Declination (J2000.0)
  41- 45  F5.2  arcsec  DEs       Declination (J2000.0)
  47- 52  F6.2  deg     GLON      Galactic longitude
  54- 59  F6.2  deg     GLAT      Galactic latitude
  61- 65  F5.3  ---     z         Redshift
      66  A1    ---   u_z         [?] uncertainty flag on z
  68- 71  F4.1  mag     Rmag      ?=0 R magnitude
  73- 96  A24   ---     Class     Source classification
  98-105  F8.1  mJy     FR        ?=0 Flux density at 1.4/0.843GHz
 107-111  I5    mJy     F143      ?=0 Flux density at 143GHz
 113-118  F6.2  fW/m2   FX        ?=0 X-ray flux 0.1-2.4keV
 120-126  E7.3  cm2/s   FF        ?=0 Fermi flux1-100GeV (in ph/cm2/s)
 128-133  F6.3  ---     aro       Spectral index radio-optical
--------------------------------------------------------------------------------

Global Notes:
Note (G1): The classifications are:
    5BZB = BL Lac objects, used for AGNs with a featureless optical
      spectrum, or having only absorption lines of galaxian origin and weak
      and narrow emission lines;
    5BZG = sources, usually reported as BL Lac objects in the literature,
      but having a spectral energy distribution (SED) with a significant
      dominance of the galaxian emission over the nuclear one
    5BZQ = Flat Spectrum Radio Quasars, with an optical spectrum showing
      broad emission lines and dominant blazar characteristics;
    BZU  = blazars of Uncertain type, adopted for a small number of sources
      having peculiar characteristics but also exhibiting blazar activity:
      for instance, occasional presence/absence of broad spectral lines or
      other features, transition objects between a radio galaxy and a BL
      Lac, galaxies hosting a low luminosity blazar Nucleus, etc.
--------------------------------------------------------------------------------

History:
    Copied at http://www.asdc.asi.it/bzcat

================================================================================
(End)                                      Patricia Vannier [CDS]    11-Feb-2016
