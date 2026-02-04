---
layout: page
title: Blog
description: Thoughts on machine learning, distributed systems, and more
permalink: /blog/
---

{% if site.posts.size > 0 %}
<ul class="posts-list">
{% for post in site.posts %}
<li class="post-item">
  <h3 class="post-item-title">
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </h3>
  <p class="post-item-meta">
    {{ post.date | date: "%B %d, %Y" }}
    {% if post.categories.size > 0 %}
    &middot;
    {% for category in post.categories %}
    <span>{{ category }}</span>{% unless forloop.last %}, {% endunless %}
    {% endfor %}
    {% endif %}
  </p>
  {% if post.excerpt %}
  <p class="post-item-excerpt">{{ post.excerpt | strip_html | truncate: 200 }}</p>
  {% endif %}
</li>
{% endfor %}
</ul>
{% else %}
<p style="color: var(--color-text-muted); text-align: center; padding: 3rem 0;">
  No posts yet. Check back soon!
</p>
{% endif %}
