document.addEventListener('DOMContentLoaded', function() {
    var elems = document.querySelectorAll('.tooltipped');
    M.Tooltip.init(elems);
});
document.addEventListener('DOMContentLoaded', function() {
    // Initialisation des tooltips Materialize
    var elems = document.querySelectorAll('.tooltipped');
    M.Tooltip.init(elems);

    // Gestion des étoiles de notation
    const starsContainers = document.querySelectorAll('.star-rating');
    
    starsContainers.forEach(container => {
        const stars = container.querySelectorAll('.star');
        const lieuId = parseInt(container.dataset.lieuId); // Conversion en nombre

        // Fonction pour mettre à jour l'affichage des étoiles
        const updateStars = (rating) => {
            stars.forEach((star, index) => {
                if (index < rating) {
                    star.classList.add('fa-star');
                    star.classList.remove('fa-star-o');
                } else {
                    star.classList.add('fa-star-o');
                    star.classList.remove('fa-star');
                }
            });
        };

        // Événements pour chaque étoile
        stars.forEach(star => {
            // Survol de la souris
            star.addEventListener('mouseover', function() {
                const value = parseInt(this.dataset.value);
                updateStars(value);
            });

            // Sortie de la souris
            star.addEventListener('mouseout', function() {
                // On réinitialise à la note actuelle (si existante)
                const currentRating = container.dataset.currentRating || 0;
                updateStars(currentRating);
            });

            // Clic
            star.addEventListener('click', async function() {
                const value = parseInt(this.dataset.value);
                
                try {
                    const response = await fetch("/noter_lieu", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                            "Accept": "application/json"
                        },
                        body: JSON.stringify({ 
                            lieu_id: lieuId,
                            note: value 
                        })
                    });

                    const data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.error || "Erreur lors de l'enregistrement");
                    }

                    // Mise à jour de l'affichage après confirmation du serveur
                    updateStars(value);
                    container.dataset.currentRating = value; // Sauvegarde de la note actuelle
                    
                } catch (err) {
                    console.error("Erreur:", err);
                    M.toast({
                        html: `Erreur: ${err.message}`,
                        classes: 'rounded red'
                    });
                }
            });
        });

        // Optionnel: Charger la note existante au chargement de la page
        const loadExistingRating = async () => {
            try {
                const response = await fetch(`/get_rating?lieu_id=${lieuId}`);
                if (response.ok) {
                    const data = await response.json();
                    if (data.rating) {
                        updateStars(data.rating);
                        container.dataset.currentRating = data.rating;
                    }
                }
            } catch (err) {
                console.log("Aucune note existante ou erreur de chargement");
            }
        };
        
        loadExistingRating();
    });
});


$(document).on('click', '.btn-delete-interaction', function() {
    const interactionId = $(this).data('interaction-id');
    const button = $(this);
    const card = button.closest('.destination-block');
    
    if (confirm("Êtes-vous sûr de vouloir supprimer cette interaction ?")) {
        button.prop('disabled', true).html('<i class="material-icons">hourglass_empty</i> Suppression...');
        
        $.ajax({
            url: '/supprimer_interaction',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ interaction_id: interactionId }),
            success: function(response) {
                if (response.success) {
                    card.fadeOut(300, function() {
                        $(this).remove();
                        if ($('.destination-block').length === 0) {
                            location.reload();
                        }
                    });
                } else {
                    alert("Erreur: " + (response.error || "Échec de la suppression"));
                    button.prop('disabled', false).html('<i class="material-icons">delete</i> Supprimer');
                }
            },
            error: function(xhr) {
                let errorMsg = "Une erreur est survenue lors de la suppression";
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMsg += ": " + xhr.responseJSON.error;
                }
                alert(errorMsg);
                button.prop('disabled', false).html('<i class="material-icons">delete</i> Supprimer');
            }
        });
    }
});


function animateOnScroll() {
    const grids = document.querySelectorAll('.lieux-grid');
    
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const grid = entry.target;
          grid.classList.add('animate');
          
          // Définit dynamiquement les délais
          const cards = grid.querySelectorAll('.destination-block');
          cards.forEach((card, index) => {
            card.style.setProperty('--delay', index % 6); // Répète tous les 6 éléments
          });
        }
      });
    }, { threshold: 0.1 });
  
    grids.forEach(grid => observer.observe(grid));
  }
  
  document.addEventListener('DOMContentLoaded', animateOnScroll);


  document.addEventListener('DOMContentLoaded', function() {
    // Initialisation des selects Materialize
    var selects = document.querySelectorAll('select');
    M.FormSelect.init(selects, {
      dropdownOptions: {
        constrainWidth: false,
        coverTrigger: false,
        alignment: 'top' // Force l'alignement vers le haut
      }
    });
  
    // Fermer les autres dropdowns quand un select est ouvert
    selects.forEach(select => {
      select.addEventListener('click', function() {
        selects.forEach(s => {
          if (s !== select) {
            const instance = M.FormSelect.getInstance(s);
            instance.close();
          }
        });
      });
    });
  });