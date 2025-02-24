---
hide:
  - date
  - navigation
  - toc
home: true
nostatistics: true
comments: false
icon: material/home
---

<br><br><br><br><br><br>

<h1 style="font-size: 50px; 
           font-weight: bold; 
           background: linear-gradient(45deg, #9999ff, #f3a683);
           -webkit-background-clip: text; 
           color: transparent;">
    Welcome to Ecank's Blog!
</h1>

<style>
  /* 打字动画 */
  @keyframes typing {
    0% { 
      clip-path: inset(0 100% 0 0); /* 从左往右逐步显示 */
    }
    100% { 
      clip-path: inset(0 0 0 0);
    }
  }

  h1 {
    overflow: hidden;
    white-space: nowrap;
    animation: typing 3s steps(30) 1s  1 normal both;
    margin: 0;          /* 移除默认外边距 */
    display: inline-block; /* 自然宽度 */
  }

  /* 父容器（如 body）简单居中 */
  body {
    text-align: center; /* 水平居中 */
    margin: 20px 0;     /* 添加基础边距 */
  }
</style>
<span style="display: block; text-align: center; font-size: 18px;">
<!--[:octicons-link-16: My friends!](./Links.md) /  -->
[:octicons-info-16: About Me](./about.md)
<!-- [:material-chart-line: Statistics](javascript:toggle_statistics();) -->
</span>


<div id="statistics" markdown="1" class="card" style="width: 27em; border-color: transparent; opacity: 0; margin-left: auto; margin-right: 0; font-size: 110%">
  <div style="padding-left: 1em;" markdown="1">
    <li>Website Operating Time: <span id="web-time"></span></li>
    <li>Total Visitors: <span id="busuanzi_value_site_uv"></span> people</li>
    <li>Total Visits: <span id="busuanzi_value_site_pv"></span> times</li>
  </div>
</div>

<script>
function updateTime() {
    var date = new Date();
    var now = date.getTime();
    var startDate = new Date("2025/02/23 00:00:00"); // 修改为你的网站开始时间
    var start = startDate.getTime();
    var diff = now - start;
    var y, d, h, m;
    y = Math.floor(diff / (365 * 24 * 3600 * 1000));
    diff -= y * 365 * 24 * 3600 * 1000;
    d = Math.floor(diff / (24 * 3600 * 1000));
    h = Math.floor(diff / (3600 * 1000) % 24);
    m = Math.floor(diff / (60 * 1000) % 60);
    if (y == 0) {
        document.getElementById("web-time").innerHTML = d + "<span> </span>d<span> </span>" + h + "<span> </span>h<span> </span>" + m + "<span> </span>m";
    } else {
        document.getElementById("web-time").innerHTML = y + "<span> </span>y<span> </span>" + d + "<span> </span>d<span> </span>" + h + "<span> </span>h<span> </span>" + m + "<span> </span>m";
    }
    setTimeout(updateTime, 1000 * 60);
}
updateTime();

function toggle_statistics() {
    var statistics = document.getElementById("statistics");
    if (statistics.style.opacity == 0) {
        statistics.style.opacity = 1;
    } else {
        statistics.style.opacity = 0;
    }
}
</script>
