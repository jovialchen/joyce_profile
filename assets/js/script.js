let currentSection = 0;
const sections = document.querySelectorAll('.section');

function scrollHandler(event) {
  event.preventDefault();
  const delta = event.deltaY;

  if (delta > 0 && currentSection < sections.length - 1) {
    currentSection++;
  } else if (delta < 0 && currentSection > 0) {
    currentSection--;
  }

  sections[currentSection].scrollIntoView({ behavior: 'smooth' });
}

window.addEventListener('wheel', scrollHandler, { passive: false });
