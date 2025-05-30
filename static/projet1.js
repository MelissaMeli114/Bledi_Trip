/*=============== SHOW HIDE PASSWORD LOGIN ===============*/
const passwordAccess = (loginPass, loginEye) =>{
    const input = document.getElementById(loginPass),
          iconEye = document.getElementById(loginEye)
 
    iconEye.addEventListener('click', () =>{
       // Change password to text
       input.type === 'password' ? input.type = 'text'
                                       : input.type = 'password'
 
       // Icon change
       iconEye.classList.toggle('ri-eye-fill')
       iconEye.classList.toggle('ri-eye-off-fill')
    })
 }
 passwordAccess('password','loginPassword')
 
 /*=============== SHOW HIDE PASSWORD CREATE ACCOUNT ===============*/
 const passwordRegister = (loginPass, loginEye) =>{
    const input = document.getElementById(loginPass),
          iconEye = document.getElementById(loginEye)
 
    iconEye.addEventListener('click', () =>{
       // Change password to text
       input.type === 'password' ? input.type = 'text'
                                       : input.type = 'password'
 
       // Icon change
       iconEye.classList.toggle('ri-eye-fill')
       iconEye.classList.toggle('ri-eye-off-fill')
    })
 }
 passwordRegister('passwordCreate','loginPasswordCreate')
 
 /*=============== SHOW HIDE LOGIN & CREATE ACCOUNT ===============*/
 const loginAcessRegister = document.getElementById('loginAccessRegister'),
       buttonRegister = document.getElementById('loginButtonRegister'),
       buttonAccess = document.getElementById('loginButtonAccess')
 
 buttonRegister.addEventListener('click', () => {
    loginAcessRegister.classList.add('active')
 })
 
 buttonAccess.addEventListener('click', () => {
    loginAcessRegister.classList.remove('active')
 })
 
// Vérifie l'URL au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
   const hash = window.location.hash;
   
   if (hash === '#register') {
       // Active la section register si l'URL contient #register
       document.getElementById('loginAccessRegister').classList.add('active');
   }
   // Sinon, la section login reste active par défaut
});

// Gestionnaire pour le bouton "Create Account"
document.getElementById('loginButtonRegister').addEventListener('click', function(e) {
   e.preventDefault();
   window.location.hash = 'register';
   document.getElementById('loginAccessRegister').classList.add('active');
});

// Gestionnaire pour le bouton "Log In"
document.getElementById('loginButtonAccess').addEventListener('click', function(e) {
   e.preventDefault();
   window.location.hash = 'login';
   document.getElementById('loginAccessRegister').classList.remove('active');
});

document.getElementById('goBackButton').addEventListener('click', function() {
   window.history.back();
});


const API_KEY = "573df8eb65077b5957de5861a32b9631"; // Remplacer par votre clé API OpenWeatherMap

// Fonction pour activer la géolocalisation et afficher les données
document.getElementById('enableLocationButton').addEventListener('click', function() {
    // Vérifier si la géolocalisation est supportée par le navigateur
    if ("geolocation" in navigator) {
        // Demander la position de l'utilisateur
        navigator.geolocation.getCurrentPosition(
            async (position) => {
                let lat = position.coords.latitude;
                let lon = position.coords.longitude;

                // Récupérer le nom du lieu (ville, pays)
                let locationName = await getLocationName(lat, lon);
                document.getElementById("position").innerText = 
                    `Latitude: ${lat}, Longitude: ${lon}\nLieu: ${locationName}`;

                // Récupérer les données météo
                let weatherData = await getWeather(lat, lon);
                document.getElementById("weather").innerText = weatherData;

                // Déterminer la saison actuelle
                let season = getSeason(lat);
                document.getElementById("season").innerText = `Saison: ${season}`;
            },
            (error) => {
                console.error("Erreur de géolocalisation: ", error.message);
                alert("Impossible d'obtenir votre position.");
            }
        );
    } else {
        console.log("La géolocalisation n'est pas supportée par ce navigateur.");
        alert("La géolocalisation n'est pas supportée.");
    }
});

// Fonction pour récupérer le nom du lieu via OpenStreetMap Nominatim
async function getLocationName(lat, lon) {
    try {
        let response = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&accept-language=fr`);
        let data = await response.json();
        
        if (data && data.address) {
            return `${data.address.city || data.address.town || data.address.village}, ${data.address.state || ""}, ${data.address.country}`;
        } else {
            return "Lieu inconnu";
        }
    } catch (error) {
        console.error("Erreur lors de la récupération du nom du lieu :", error);
        return "Erreur de récupération du lieu";
    }
}

// Fonction pour récupérer la météo via OpenWeatherMap
async function getWeather(lat, lon) {
    try {
        let response = await fetch(`https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${API_KEY}&units=metric&lang=fr`);
        let data = await response.json();

        if (data && data.weather) {
            let description = data.weather[0].description;
            let temperature = data.main.temp;
            return `Météo: ${description}, Température: ${temperature}°C`;
        } else {
            return "Données météo indisponibles";
        }
    } catch (error) {
        console.error("Erreur lors de la récupération des données météo :", error);
        return "Erreur de récupération de la météo";
    }
}

// Fonction pour déterminer la saison en fonction de la latitude et du mois actuel
function getSeason(lat) {
    let month = new Date().getMonth() + 1; // Mois actuel (1 = Janvier, 12 = Décembre)

    if (lat >= 0) { // Hémisphère Nord
        if (month >= 3 && month <= 5) return "Printemps 🌸";
        if (month >= 6 && month <= 8) return "Été ☀️";
        if (month >= 9 && month <= 11) return "Automne 🍂";
        return "Hiver ❄️"; // Décembre - Février
    } else { // Hémisphère Sud
        if (month >= 3 && month <= 5) return "Automne 🍂";
        if (month >= 6 && month <= 8) return "Hiver ❄️";
        if (month >= 9 && month <= 11) return "Printemps 🌸";
        return "Été ☀️"; // Décembre - Février
    }
}