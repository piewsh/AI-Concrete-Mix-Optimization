document.addEventListener("DOMContentLoaded", function() {
    // Highlight active navigation link
    const navLinks = document.querySelectorAll(".nav-link");
    navLinks.forEach(link => {
        link.addEventListener("click", function() {
            navLinks.forEach(l => l.classList.remove("active"));
            this.classList.add("active");
        });
    });
});


