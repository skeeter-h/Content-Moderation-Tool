// Add an event listener for the form submission
document.getElementById('tweetForm').addEventListener('submit', function(event) {
    // Prevent the default form submission behavior
    event.preventDefault();

    // Call the submitTweet function
    submitTweet();
});


function submitTweet() {
    const tweetInput = document.getElementById('tweetInput').value;

    // Send a POST request to the Flask server
    fetch('/evaluate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: tweetInput,
        }),
    })
    .then(response => response.json())
    .then(data => {
        // Log the data and its type for debugging purposes
        console.log('Server Response:', data);
        console.log('Type of data.result:', typeof data.result);

        // Handle the response from the server (data.result)
        if (parseInt(data.result, 10) === 0) {
            // Add the new post to the feed
            addPostToFeed(tweetInput, false);

            // Alert for non-harmful message (you can customize this)
            alert('Message is not harmful');
        } else {
            // Display a warning message (you can customize this)
            alert('Warning: This tweet may contain harmful content');

            // Add the new post to the feed with a warning banner
            addPostToFeed(tweetInput, true);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}



function addPostToFeed(message, isHarmful) {
    // Create a new post element
    const newPost = document.createElement('div');
    newPost.className = 'post';

    // Structure the post HTML (customize as needed)
    newPost.innerHTML = `
        <div class="post__avatar">
            <img src="https://i.pinimg.com/originals/a6/58/32/a65832155622ac173337874f02b218fb.png" alt="" />
        </div>
        <div class="post__body">
            <div class="post__header">
                <div class="post__headerText">
                    <h3>Skeeter Hrisca<span class="post__headerSpecial"
                    ><span class="material-icons post__badge"> verified </span>@skeets</span
                  ></h3>
                </div>
                <div class="post__headerDescription">
                    <p>${message}</p>
                </div>
            </div>
            <!-- Add other post content here -->
            <div class="post__footer">
                <span class="material-icons"> repeat </span>
                <span class="material-icons"> favorite_border </span>
                <span class="material-icons"> publish </span>
            </div>
        </div>
    `;

    // Check if the tweet is harmful and add a warning banner if necessary
    if (isHarmful) {
        const warningBanner = document.createElement('div');
        warningBanner.className = 'post__warning';
        warningBanner.textContent = 'Warning: This tweet may contain harmful content';

        newPost.appendChild(warningBanner);
    }

    // Get the tweetBox element and insert the new post below it
    const tweetBox = document.querySelector('.tweetBox');
    tweetBox.parentNode.insertBefore(newPost, tweetBox.nextSibling);

    // Clear the tweet input after submitting
    document.getElementById('tweetInput').value = '';
}


