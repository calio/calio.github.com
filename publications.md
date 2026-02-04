---
layout: page
title: Publications
description: Research papers and preprints
permalink: /publications/
---

<ul class="publications-list">
{% for pub in site.data.publications %}
<li class="publication-item">
  {% if pub.image %}
  <img src="{{ pub.image }}" alt="{{ pub.title }}" class="publication-image">
  {% endif %}
  <div class="publication-content">
    <h3 class="publication-title">
      {% if pub.links[0] %}
      <a href="{{ pub.links[0].url }}" target="_blank" rel="noopener">{{ pub.title }}</a>
      {% else %}
      {{ pub.title }}
      {% endif %}
    </h3>
    <p class="publication-authors">{{ pub.authors }}</p>
    <p class="publication-venue">{{ pub.venue }}, {{ pub.year }}</p>
    {% if pub.links %}
    <div class="publication-links">
      {% for link in pub.links %}
      <a href="{{ link.url }}" target="_blank" rel="noopener">{{ link.name }}</a>
      {% endfor %}
    </div>
    {% endif %}
  </div>
</li>
{% endfor %}
</ul>

For a complete list of publications, see my [Google Scholar profile](https://scholar.google.com/citations?user=Jkn9ksIAAAAJ).
