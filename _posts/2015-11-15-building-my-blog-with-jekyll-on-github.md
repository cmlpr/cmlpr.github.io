---
layout: post
title: "Building my blog with Jekyll on Github by CMLPR"
date: 2015-11-15
---

This is my first post in my Github based blog. I have no blogging experience but I'll do my best and share things about I like as much as I can.

<!--more-->

I created this blog following the Github Pages. If you would like to do the same, you can start from <a href="https://pages.github.com" target="_blank">here</a>. In order to see my blog locally, I also installed all the components such as `Ruby`, `RubyGems`, `Bundler` and `Jekyll` on my computer. The instructions are clearly described in Github Pages link. I saw many nice blogs using `Jekyll-Bootstrap (JB)` and wanted to build my blog on top of it too. So as a next step I cloned the files and read the documentation from this <a href="http://jekyllrb.com" target="_blank"> link</a>. After a few small modifications, I pushed my local files and folders to Github.

Files that I modified initially are:

* _config.yml
  - update author, title, tagline ...
  - add `rdiscount` markdown and extensions
  - add custom `excerpt_separator` for summaries
  - add github link to `production_url`
  - choose comments & analytics providers and update information
* index.md
  - add a section for recently published posts


Jekyll comes with code highlighting functionality. To use it, first make sure _config.yml contains this line:

```
highlighter: pygments
```

To highlight `html` code, I generated the `css` file and placed it in my current theme's asset folder. You need `Python` and `Pygments` module to perform this.

```
python -m pip install Pygments
pygmentize -S default -f html > assets/themes/bootstrap-3/css/pygments/default.css
```

And add a link to the css file in the `_includes/themes/bootsrap-3/defaults.html` file.

An example highlighted code snippet looks like this:

{% highlight HTML linenos %}
<!DOCTYPE html>
<html>
    <body>
        <h1>Heading</h1>
        <!-- This is a comment -->
        <p>Paragraph</p>
        <a href="url">link text</a>
</body>
</html>
{% endhighlight %}

Another useful functionality is the ability to use math blocks in posts. Activating is exteremely easiy.

Add kramdown option to `_config.yml`:

```
kramdown:
  input: GFM
```

And add MathJax script into `<head>` section of `_includes/themes/bootsrap-3/defaults.html`.

{% highlight HTML %}
<!-- MathJax -->
    <script type="text/javascript"
    src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
{% endhighlight %}

Refer to [MathJax documentation](http://docs.mathjax.org/en/latest/start.html "MathJax Documentation") for more information and examples.

\\[ \mathbf{X} = \mathbf{A^\mathsf{T}} \mathbf{B} \\]
\\[ \int_{0}^{\infty}f(x)\,\mathrm{d}x \\]

I would like to keep this first post short so that I can focus on the modifications.

Cmlpr
