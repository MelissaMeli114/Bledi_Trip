const editBtn = document.getElementById('edit-btn');
const form = document.getElementById('profile-form');
const inputs = document.querySelectorAll('#profile-form input');

let isEditing = false;

editBtn.addEventListener('click', (e) => {
    e.preventDefault(); // empêche le rechargement ou l'envoi du formulaire

    if (!isEditing) {
        // On passe en mode édition
        inputs.forEach(input => {
            if (input.name !== "" && input.name !== "visited_places" && input.name !=="pseudo") {
                input.disabled = false;
            }
        });
        editBtn.innerHTML = '<i class="fa fa-floppy-o" aria-hidden="true"></i> Enregistrer';
        isEditing = true;
    } else {
        // On soumet le formulaire pour sauvegarder les modifications
        form.submit();
    }
});

