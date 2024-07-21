#import "@preview/touying:0.4.0": *

// Themes: default, simple, metropolis, dewdrop, university, aqua
#let s = themes.dewdrop.register(aspect-ratio: "16-9", navigation: none)
#let s = (s.methods.info)(
  self: s,
  title: [Introduction to Machine Learning in Python],
  author: [Presented by Nicholas Miklaucic],
  date: datetime.today(),
)
#let (init, slides, touying-outline, alert) = utils.methods(s)
#show: init

#import "@preview/cetz:0.2.2": canvas, plot


#set text(size: 16pt, font: "Source Sans Pro", weight: "regular")
#show quote: set text(size: 12pt, weight: "regular")
#show quote: set par(first-line-indent: 2em)
#show quote: set block(above: 1em, below: 0em)
#set quote(block: true)
#show figure.caption: set text(size: 12pt, weight: "regular", style: "italic")
#show heading.where(level: 3): set text(weight: "semibold", fill: blue)

#let (slide, empty-slide, title-slide, focus-slide) = utils.slides(s)

#show: slides.with(slide-level: 1)


= Introduction: What is Machine Learning?

== Machine Learning
Making computers learn how to do things. More formally:

#quote([A computer program is said to learn from experience E with respect to some class of tasks T, and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.], attribution: [Tom Mitchell, *Machine Learning*])
