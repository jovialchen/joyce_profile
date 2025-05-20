    const gifArrowBtn = document.getElementById("gifArrowBtn");
    const gifArrow = document.getElementById("gifArrow");
    const progressBar = document.getElementById("progressBar");

    let isActive = false;

    window.onscroll = function () {
      scrollFunction();
      updateProgressBar();
    };

    function scrollFunction() {
    if (document.documentElement.scrollTop > 200) {
        // 用 flex 而不是 block
        gifArrowBtn.style.display = "flex";
    } else {
        gifArrowBtn.style.display = "none";
    }
    }


    function updateProgressBar() {
      const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
      const scrollHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrollPercent = (scrollTop / scrollHeight) * 100;
      progressBar.style.width = scrollPercent + "%";
    }

    gifArrowBtn.addEventListener("click", () => {
      isActive = !isActive;
      gifArrow.src = isActive
        ? "assets/images/arrow-active.gif"
        : "assets/images/arrow-default.gif";

      gifArrowBtn.classList.toggle("active");

      window.scrollTo({ top: 0, behavior: "smooth" });
    });