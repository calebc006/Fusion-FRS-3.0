const select = document.getElementById('stream_src_select');
const customInput = document.getElementById('stream_src_custom');
const form = document.getElementById('init');

select.addEventListener('change', function () {
  if (this.value === 'custom') {
    // Hide the select's name to prevent duplicate keys
    select.removeAttribute('name');

    // Show and enable the custom input
    customInput.style.display = 'block';
    customInput.setAttribute('required', 'required');
    customInput.setAttribute('name', 'stream_src');
  } else {
    // Restore name to select
    select.setAttribute('name', 'stream_src');

    // Hide and disable the custom input
    customInput.style.display = 'none';
    customInput.removeAttribute('required');
    customInput.removeAttribute('name');
    customInput.value = ''; // Clear out stale value
  }
});

// Optional: Form validation reminder
form.addEventListener('submit', function (e) {
  if (select.value === 'custom' && !customInput.value.trim()) {
    e.preventDefault();
    alert('Please enter a valid custom RTSP URL.');
  }
});
