document.addEventListener('DOMContentLoaded', function() {
    const wrapper = document.getElementById('videoWrapper');
    const video = document.getElementById('pvVideo');
    const trigger = document.getElementById('playTrigger');

    trigger.addEventListener('click', function() {
        video.play();
        wrapper.classList.add('is-playing');
        video.controls = true;
    });

    video.addEventListener('ended', function() {
        wrapper.classList.remove('is-playing');
        video.controls = false;
        video.currentTime = 0;
    });
});