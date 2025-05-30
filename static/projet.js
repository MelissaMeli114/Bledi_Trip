function changeBg(bg, title){
    const banner = document.querySelector('.banner');
    const contents = document.querySelectorAll('.content');
    banner.style.background=`url('/static/images/${bg}')`;
    banner.style.backgroundSize='cover';
    banner.style.backgroundPosition='center';

    contents.forEach(content => {
        content.classList.remove('active');
        if(content.classList.contains(title)){
            content.classList.add('active');
        }
    });
}

window.addEventListener('scroll', function () {
    const header = document.querySelector('header');
    if (window.scrollY > 50) {
      header.classList.add('scrolled');
    } else {
      header.classList.remove('scrolled');
    }
  });

  const links = document.querySelectorAll('.nav li a');

  links.forEach(link => {
    link.addEventListener('click', () => {
      links.forEach(l => l.classList.remove('active'));
      link.classList.add('active');
    });
  });

  // Variable pour suivre l'état des animations
let aboutAnimationsPlayed = false;

// Fonction améliorée pour activer les animations About
function activateAboutAnimations() {
  if (aboutAnimationsPlayed) return;
  
  const aboutSection = document.getElementById('about');
  if (!aboutSection) return;

  // Elements à animer
  const elements = {
    title: aboutSection.querySelector('.about_title'),
    desc: aboutSection.querySelector('.about.description'),
    button: aboutSection.querySelector('#about .button a'),
    img1: aboutSection.querySelector('.image_one'),
    img2: aboutSection.querySelector('.image_two')
  };

  // Vérification que tous les éléments existent
  if (!elements.title || !elements.desc || !elements.button || !elements.img1 || !elements.img2) return;

  // Reset des animations
  Object.values(elements).forEach(el => {
    el.style.animation = 'none';
    el.style.opacity = '0';
  });

  // Force le reflow
  void aboutSection.offsetWidth;

  // Applique les animations avec les délais
  elements.title.style.animation = 'fadeDown 1s ease forwards';
  elements.desc.style.animation = 'fadeLeft 1s ease forwards 0.2s';
  elements.button.style.animation = 'floatInAir 1s ease forwards 0.4s, floatLoop 3s ease-in-out infinite 1s';
  elements.img1.style.animation = 'zoomInSlide 1s ease forwards 0.6s';
  elements.img2.style.animation = 'zoomInSlide 1s ease forwards 0.8s';

  aboutAnimationsPlayed = true;
}

// Reset des animations quand la section quitte la vue
function resetAboutAnimations() {
  aboutAnimationsPlayed = false;
}

// Observer pour le scroll
const aboutObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      activateAboutAnimations();
    } else {
      resetAboutAnimations();
    }
  });
}, { threshold: 0.1 });

// Activation de l'observer
const aboutSection = document.getElementById('about');
if (aboutSection) {
  aboutObserver.observe(aboutSection);
}

// Gestion améliorée du clic navbar
document.querySelectorAll('.nav a[href="#about"]').forEach(link => {
  link.addEventListener('click', function(e) {
    e.preventDefault();
    
    // Vérifie si on est déjà dans la section About
    const isAlreadyInAbout = window.location.hash === '#about';
    
    // Scroll vers la section
    aboutSection.scrollIntoView({ behavior: 'smooth' });
    
    // Active les animations seulement si on n'était pas déjà dans la section
    if (!isAlreadyInAbout) {
      setTimeout(() => {
        activateAboutAnimations();
      }, 800);
    }
  });
});

// Variable pour suivre l'état des animations
let featuresAnimationsPlayed = false;

// Fonction pour activer les animations de la section Features
function activateFeaturesAnimations() {
  if (featuresAnimationsPlayed) return;

  const featuresSection = document.getElementById('features');
  if (!featuresSection) return;

  const featuresTitle = featuresSection.querySelector('.features-title');
  const featureBoxes = featuresSection.querySelectorAll('.feature-box');

  if (!featuresTitle || featureBoxes.length < 3) return;

  // Reset animation + opacity
  featuresTitle.style.animation = 'none';
  featuresTitle.style.opacity = '0';
  featureBoxes.forEach(box => {
    box.style.animation = 'none';
    box.style.opacity = '0';
  });

  // Force reflow
  void featuresSection.offsetWidth;

  // Appliquer les animations avec décalages
  featuresTitle.style.animation = 'slideDownFadeIn 1s ease-out forwards 0.3s';
  featureBoxes[0].style.animation = 'fadeInUp 0.8s ease forwards 0.2s';
  featureBoxes[1].style.animation = 'fadeInUp 0.8s ease forwards 0.5s';
  featureBoxes[2].style.animation = 'fadeInUp 0.8s ease forwards 0.9s';

  featuresAnimationsPlayed = true;
}

// Reset des animations lorsque la section sort de la vue
function resetFeaturesAnimations() {
  featuresAnimationsPlayed = false;
}

// Observer pour déclencher au scroll
const featuresObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      activateFeaturesAnimations();
    } else {
      resetFeaturesAnimations();
    }
  });
}, { threshold: 0.35 });

// Activation de l'observer
const featuresSection = document.getElementById('features');
if (featuresSection) {
  featuresObserver.observe(featuresSection);
}

// Gestion du clic navbar pour #features
document.querySelectorAll('.nav a[href="#features"]').forEach(link => {
  link.addEventListener('click', function(e) {
    e.preventDefault();

    const isAlreadyInFeatures = window.location.hash === '#features';

    // Scroll vers la section
    featuresSection.scrollIntoView({ behavior: 'smooth' });

    if (!isAlreadyInFeatures) {
      setTimeout(() => {
        activateFeaturesAnimations();
      }, 800); // Attendre que le scroll se termine
    }
  });
});

let contactAnimationPlayed = false;

function activateContactAnimation() {
  if (contactAnimationPlayed) return;

  const footerContainer = document.querySelector('.footer-container');
  if (!footerContainer) return;

  // Reset animation + visibilité
  footerContainer.style.animation = 'none';
  footerContainer.style.opacity = '0';
  footerContainer.style.transform = 'translateY(30px)';

  // Force reflow
  void footerContainer.offsetWidth;

  // Appliquer animation
  footerContainer.style.animation = 'fadeSlideUp 1s ease-out forwards';
  footerContainer.style.animationDelay = '0.2s';

  contactAnimationPlayed = true;
}

function resetContactAnimation() {
  contactAnimationPlayed = false;
}

// Observer pour scroll
const contactSection = document.getElementById('contact');
if (contactSection) {
  const contactObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        activateContactAnimation();
      } else {
        resetContactAnimation();
      }
    });
  }, { threshold: 0.1 }); // Tu peux ajuster le seuil ici

  contactObserver.observe(contactSection);
}

// Clic sur navbar
document.querySelectorAll('.nav a[href="#contact"]').forEach(link => {
  link.addEventListener('click', function(e) {
    e.preventDefault();

    const isAlreadyInContact = window.location.hash === '#contact';

    contactSection.scrollIntoView({ behavior: 'smooth' });

    if (!isAlreadyInContact) {
      setTimeout(() => {
        activateContactAnimation();
      }, 800); // après scroll smooth
    }
  });
});

// Variables de contrôle pour Home
let homeAnimationsPlayed = false;

// Fonction pour activer les animations Home
function activateHomeAnimations() {
  if (homeAnimationsPlayed) return;
  
  const homeSection = document.getElementById('home');
  if (!homeSection) return;

  // Éléments à animer
  const elements = {
    logo: document.querySelector('.logo'),
    navItems: document.querySelectorAll('.nav li'),
    accountItems: document.querySelectorAll('.compte li'),
    bannerContent: homeSection.querySelector('.content.active'),
    carousel: homeSection.querySelector('.carousel-box'),
    sci: homeSection.querySelectorAll('.sci a')
  };

  // Reset des animations
  elements.logo.style.animation = 'none';
  elements.navItems.forEach(item => item.style.animation = 'none');
  elements.accountItems.forEach(item => item.style.animation = 'none');
  if (elements.bannerContent) {
    const bannerElements = {
      h4: elements.bannerContent.querySelector('h4'),
      p: elements.bannerContent.querySelector('p'),
      button: elements.bannerContent.querySelector('.button')
    };
    Object.values(bannerElements).forEach(el => el && (el.style.animation = 'none'));
  }
  elements.carousel.style.animation = 'none';
  elements.sci.forEach(item => item.style.animation = 'none');
  
  // Force le reflow
  void homeSection.offsetWidth;

  // Applique les animations avec les délais
  elements.logo.style.animation = 'fadeInLogo 1s ease forwards 0.5s';
  
  elements.navItems.forEach((item, index) => {
    item.style.animation = `fadeInTop 0.6s ease forwards ${0.3 + index * 0.1}s`;
  });
  
  elements.accountItems.forEach((item, index) => {
    item.style.animation = `fadeInTop 0.6s ease forwards ${0.7 + index * 0.1}s`;
  });

  if (elements.bannerContent) {
    const bannerElements = {
      h4: elements.bannerContent.querySelector('h4'),
      p: elements.bannerContent.querySelector('p'),
      button: elements.bannerContent.querySelector('.button')
    };
    if (bannerElements.h4) bannerElements.h4.style.animation = 'fadeInUp 0.8s ease forwards 0.2s';
    if (bannerElements.p) bannerElements.p.style.animation = 'fadeInUp 0.8s ease forwards 0.4s';
    if (bannerElements.button) bannerElements.button.style.animation = 'fadeInUp 0.8s ease forwards 0.6s';
  }

  elements.carousel.style.animation = 'fadeUp 1.5s ease-out forwards 0.8s';
  
  elements.sci.forEach((item, index) => {
    item.style.animation = `fadeInFloat 1s ease forwards ${1 + index * 0.2}s`;
  });

  homeAnimationsPlayed = true;
}

// Reset des animations Home
function resetHomeAnimations() {
  homeAnimationsPlayed = false;
}

// Observer pour Home
const homeObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      activateHomeAnimations();
    } else {
      resetHomeAnimations();
    }
  });
}, { threshold: 0.1 });

// Activation de l'observer
const homeSection = document.getElementById('home');
if (homeSection) {
  homeObserver.observe(homeSection);
}

// Gestion du clic navbar pour Home
document.querySelectorAll('.nav a[href="#home"]').forEach(link => {
  link.addEventListener('click', function(e) {
    e.preventDefault();
    
    const isAlreadyInHome = window.location.hash === '#home';
    window.scrollTo({ top: 0, behavior: 'smooth' });
    
    if (!isAlreadyInHome) {
      setTimeout(() => {
        activateHomeAnimations();
      }, 800);
    }
  });
});

document.addEventListener('DOMContentLoaded', function() {
  const menuToggle = document.querySelector('.menu-toggle');
  const nav = document.querySelector('.nav');
  const compte = document.querySelector('.compte');
  
  function checkScreenSize() {
      if (window.innerWidth <= 768) {
          // Mode mobile
          menuToggle.style.display = 'block';
          nav.style.display = 'none';
          compte.style.display = 'none';
      } else {
          // Mode desktop
          menuToggle.style.display = 'none';
          nav.style.display = 'flex';
          compte.style.display = 'flex';
          nav.classList.remove('active');
          compte.classList.remove('active');
      }
  }
  
  // Vérifier la taille au chargement
  checkScreenSize();
  
  // Gérer le clic sur le menu toggle
  menuToggle.addEventListener('click', function() {
      this.classList.toggle('active');
      if (this.classList.contains('active')) {
          nav.style.display = 'flex';
          compte.style.display = 'flex';
          setTimeout(() => {
              nav.classList.add('active');
              compte.classList.add('active');
          }, 10);
      } else {
          nav.classList.remove('active');
          compte.classList.remove('active');
          setTimeout(() => {
              nav.style.display = 'none';
              compte.style.display = 'none';
          }, 500);
      }
  });
  
  // Gérer le redimensionnement
  window.addEventListener('resize', function() {
      checkScreenSize();
  });
  
  // Fermer le menu au clic sur un lien (mobile)
  document.querySelectorAll('.nav a, .compte a').forEach(link => {
      link.addEventListener('click', function() {
          if (window.innerWidth <= 768) {
              menuToggle.classList.remove('active');
              nav.classList.remove('active');
              compte.classList.remove('active');
              setTimeout(() => {
                  nav.style.display = 'none';
                  compte.style.display = 'none';
              }, 500);
          }
      });
  });
});