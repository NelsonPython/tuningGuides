**Version 2.4**

Welcome to the full test template for the github md import feature of AEM. This document covers all the styles availalbe and how to implement.
Note: when loading github url into your page, it may take a number of tries to get the content to refresh. I am working with IT to resolve the bug. If you have just done an update in GitHub, it's best to wait for a few minutes to make sure you sync the latest version.

## Headers (H2)
Make sure you don't use the single # for any headers. Only the title of your article should be an H1. This isn't just a visual thing, this will impact your SEO score in google.
Always start with the ##.

### Sub header level 3 (H3)
#### Sub header level 4 (H4)
##### Sub header Level 5 (rarely used) 

<br><br>
***

## General text styling
```markdown
*This text will be italic*
_This will also be italic_

**This text will be bold**
__This will also be bold__

_You **can** combine them_
```

*This text will be italic*
_This will also be italic_

**This text will be bold**
__This will also be bold__

_You **can** combine them_

Superscript and Subscript doesn't seem to work at the moment.
```markdown
Superscript
H~2~O

Subscript
X^2^
```

Superscript example: H~2~O

Subscript example: X^2^

:bulb: 🖥️ :d

<br><br>
***
### Links
```markdown
[Intel](https://www.intel.com).
```
Example:
My favorite company is [Intel](https://www.intel.com).

mailto:
(note for spam reasons, we discourage use of email links)
[example@gitlab.com](mailto:example@gitlab.com)


<br><br>
***
### Block quotes
This doesn't work with our template. We have requested some code updates so that block quotes actually render properly on articles. I will update this file once that is done.

```markdown
> We're living the future so
> the present is our past.
```

> We're living the future so
> the present is our past.
> 
<br><br>
***
### CSS Attribution - Requested

Another Item we are exploring is the ability to call the css that is used on the site. This isn't working yet. I will update once we have the functionality in place. There are two options IT is exploring. 
We have requested that IT add this feature so we can call certain css properties into markdown.

{: .greyHighlight}
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.


<br><br>
***
### Footnotes 
Footnotes aren't working as expected. I have raised a ticket with IT to see if they can enable this feature.
```markdown
Here's a sentence with a footnote. [^1]  
  
[^1]: This is the footnote.
```
Here's a sentence with a footnote. [^2]  
  
[^2]: This is another footnote to go with the first.

<br><br>
***
## Examples of math in .md

This expression $\sum_{i=1}^n X_i$ is inlined but doesn't work at the moment.

When this is a full expression, it works fine.
$$
\Gamma(z) = \int_0^\infty t^{z-1}e^{-t}dt\,.
$$


<br><br>
***
## Code on your page
Adding code into your sentence is simple. 
```markdown
`this is your code`
```

Example
Some `inline code` if you need to put inside a sentence.


If you have javascript:
```javascript
// An highlighted block
var foo = 'bar';
```

A very common one on the DevZone is bash
```bash
export I_MPI_ROOT=/opt/intel/oneapi/lib/intel64
export PATH=${I_MPI_ROOT}/libfabric/bin:${I_MPI_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${I_MPI_ROOT}/libfabric:${I_MPI_ROOT}:$LD_LIBRARY_PATH
export FI_PROVIDER_PATH=${I_MPI_ROOT}/libfabric
```

<details>
  <summary>Expand to see the full list of available skins</summary>
  <br>

* plaintext
* abap
* actionscript
* apacheconf
* applescript
* aspnet
* bash
* basic
* c
* coffeescript
* cpp
* csharp
* css
* d
* dart
* diff
* docker
* erlang
* fortran
* fsharp
* git
* go
* groovy
* haskell
* html
* http
* ini
* java
* javascript
* lua
* makefile
* markdown
* matlab
* nginx
* objectivec
* pascal
* perl
* php
* prolog
* python
* puppet
* r
* ruby
* rust
* sas
* scala
* scheme
* sql
* swift
* twig
* vim
* xmlxhtml
* yaml
</details>


<br><br>
***
## Lists
### Creating an ordered list
1. First item  
2. Second item  
3. Third item  
4. Fourth item

This will also do the same thing
1. First item  
1. Second item  
1. Third item  
1. Fourth item

This will also do the same thing
1. First item  
8. Second item  
3. Third item  
5. Fourth item

Most of the time a MD editor will try to fix your list numbering
You can also indent by adding a few spaces.


An example of a horizontal rule
```markdown
***
```
***


<br><br>
***
### Creating Unordered Lists
- First item  
- Second item  
- Third item  
- Fourth item

**Split Lists**
- list one - item 1
- list one - item 2
     - sub item 1
     - sub item 2
- list one - item 3
<br><br>
- list two - item A
- list two - item B


<br><br>
***
## Tables

```markdown
| Syntax | Description |  
| ----------- | ----------- |  
| Header | Title |  
| Paragraph | Text |
```

| Syntax | Description |  
| ----------- | ----------- |  
| Header | Title |  
| Paragraph | Text |


<br><br>
***
## A collapsible section with markdown
This does work within the article template. THe arrow is a bit large, but I will see if there is a way to get it updated.
Code:
```markdown
<details>
  <summary>Click to expand!</summary>
  <br>
  1. A numbered
  2. list
     * With some
     * Sub bullets
</details>
```
Example
<details>
  <summary>Click to expand!</summary>
  <br>
  1. A numbered
  2. list
     * With some
     * Sub bullets
</details>


<br><br>
***

## Images

To include an image from intel.com, you can do a relative link. Make sure you put all stock imagery, marketing imagery, logos, and photos of people in intel.com where we monitor licensing and expiry dates. You can use the relative path to your image.

```markdown
![This is your Alt Text](/content/dam/www/central-libraries/us/en/images/oneapi-kits20211-4x3-rwd.png)
```

![This is your Alt Text](/content/dam/www/central-libraries/us/en/images/oneapi-kits20211-4x3-rwd.png)

<br>
You can also choose to host your screenshots, diagrams, terminal window images in your repo. Just remember, you are now supporting the live site. Don't move or delete images without updating your article. Also, make sure to use the full github URL and not a relative path.

![This is your Alt Text](https://raw.githubusercontent.com/tracyjohnsonidz/devzone-articles/main/diagram-full-workflow-16x9.webp)

<br><br>
code graphis are not available for IDZ articles.
```plantuml
!define ICONURL https://raw.githubusercontent.com/tupadr3/plantuml-icon-font-sprites/v2.1.0
skinparam defaultTextAlignment center
!include ICONURL/common.puml
!include ICONURL/font-awesome-5/gitlab.puml
!include ICONURL/font-awesome-5/java.puml
!include ICONURL/font-awesome-5/rocket.puml
!include ICONURL/font-awesome/newspaper_o.puml
FA_NEWSPAPER_O(news,good news!,node) #White {
FA5_GITLAB(gitlab,GitLab.com,node) #White
FA5_JAVA(java,PlantUML,node) #White
FA5_ROCKET(rocket,Integrated,node) #White
}
gitlab ..> java
java ..> rocket
```
<br><br>
***
## Videos
Go to the youtube video and copy the embed code. Just replace the iframe src url with your youtube video URL.

```markdown
<div>
  <div style="position:relative;padding-top:56.25%;">
    <iframe src="https://www.youtube.com/embed/c7st0drv54U" frameborder="0" allowfullscreen
      style="position:absolute;top:0;left:0;width:100%;height:100%;"></iframe>
  </div>
</div>
```

<div>
  <div style="position:relative;padding-top:56.25%;">
    <iframe src="https://www.youtube.com/embed/c7st0drv54U" frameborder="0" allowfullscreen
      style="position:absolute;top:0;left:0;width:100%;height:100%;"></iframe>
  </div>
</div>

<br>

You can also use this embed code for brightcove videos. Just replace the videoid= # in the embed code below
To find the video ID, simply right click on the video on developer.intel.com and select **Player Information**. Video ID value is listed under Source.

```markdown
<div style="position: relative; display: block; max-width: 900px;">
    <div style="padding-top: 56.25%;">
      <iframe src="https://players.brightcove.net/740838651001/default_default/index.html?videoId=6286027295001" allowfullscreen="" allow="encrypted-media" style="position: absolute; top: 0px; right: 0px; bottom: 0px; left: 0px; width: 100%; height: 100%;"></iframe>
  </div>
</div>
```

<div style="position: relative; display: block; max-width: 900px;">
  <div style="padding-top: 56.25%;">
    <iframe src="https://players.brightcove.net/740838651001/default_default/index.html?videoId=6286027295001" allowfullscreen="" allow="encrypted-media" style="position: absolute; top: 0px; right: 0px; bottom: 0px; left: 0px; width: 100%; height: 100%;"></iframe>
  </div>
</div>

multiple videos being added
<!-- blank line -->
<figure class="video_container">
  <iframe src="https://drive.google.com/file/d/0B6m34D8cFdpMZndKTlBRU0tmczg/preview" frameborder="0" allowfullscreen="true"> </iframe>
</figure>

<figure class="video_container">
  <iframe src="https://drive.google.com/file/d/0B6m34D8cFdpMZndKTlBRU0tmczg/preview" frameborder="0" allowfullscreen="true"> </iframe>
</figure>

<figure class="video_container">
  <iframe src="https://drive.google.com/file/d/0B6m34D8cFdpMZndKTlBRU0tmczg/preview" frameborder="0" allowfullscreen="true"> </iframe>
</figure>
<!-- blank line -->

this is an anchor
{: #hello-world}
```markdown
{: #hello-world}

```
**Note:** a note is something that needs to be mentioned but is apart from the context.
{: .note}

Can we embed code from github, here is a gitlab test.
<!-- leave a blank line here -->
<script src="https://gitlab.com/gitlab-org/gitlab-ce/snippets/1717978.js"></script>
<!-- leave a blank line here -->

{::options parse_block_html="false" /}

<div class="center">

<blockquote class="twitter-tweet" data-partner="tweetdeck"><p lang="en" dir="ltr">Thanks to <a href="https://twitter.com/gitlab">@gitlab</a> for joining <a href="https://twitter.com/RailsGirlsCluj">@RailsGirlsCluj</a>! <a href="https://t.co/NOoiqDWKVY">pic.twitter.com/NOoiqDWKVY</a></p>&mdash; RailsGirlsCluj (@RailsGirlsCluj) <a href="https://twitter.com/RailsGirlsCluj/status/784847271645028352">October 8, 2016</a></blockquote>
<script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>

</div>
