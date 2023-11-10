<script>
  // Data
  import dataset from "./assets/data.json";
  import staticImg from "./assets/imgs/105750338276836.jpg";
  // Define constants
  const boroughs = [
    "Staten Island",
    "Queens",
    "Manhattan",
    "Bronx",
    "Brooklyn",
  ].sort((a, b) => (a > b ? 1 : -1));
  const colors = [
    "btn-primary",
    "btn-info",
    "btn-success",
    "btn-error",
    "btn-warning",
  ];
  // State variables
  let page = "welcome";
  let img = staticImg;
  let label = "Bronx";
  let loading = false;
  let score = 0;
  let count = 0;

  // Actions
  const randInt = (N) => Math.floor(Math.random() * N);

  const getImg = () => {
    // Load dataset
    const data = dataset?.data || [];
    if (!data.length) {
      console.error("No data is available");
      return;
    }

    // Get datapoint
    const datapoint = data[randInt(data.length)];
    console.log(datapoint);

    // Save the current label
    label = datapoint.borough;
    if (!label) {
      console.error("Loaded datapoint has no label!");
    }

    // Try to serve image from the url
    let url = datapoint?.url;
    if (!url) {
      // TODO: serve image ourselves
      console.error("Not implmented");
    }

    // Set image
    img = url;
  };

  const formatChoice = (choice) => choice.toLowerCase().replaceAll(/\s/g, "_");

  const postChoice = (choice) => {
    // Increase the count (& score if correct)
    count += 1;
    if (formatChoice(choice) === formatChoice(label)) {
      score += 1;
    }

    // TODO: post score to API (data collection)
  };

  const choose = (label) => {
    // Disable buttons
    loading = true;

    // Post choice
    postChoice(label);

    // Get new image
    getImg();

    // Enable buttons
    loading = false;
  };
</script>

<div class="w-100 h-screen flex flex-col items-center">
  <h1 class="text-6xl p-8">RecogNYCe</h1>
  {#if count === 0}
    <h2 class="text-3xl italic">How well do you know New York City?</h2>
  {:else}
    <h2 class="text-3xl italic">Score: {score} / {count}</h2>
  {/if}
  <div class="flex flex-1 flex-col justify-center items-center">
    {#if page === "welcome"}
      <button
        class="btn btn-lg self-center text-5xl"
        on:click={() => (page = "game")}
      >
        START
      </button>
    {:else if page === "game"}
      <div class="card w-4/5 bg-base-200 shadow-xl" style="max-width: 600px;">
        <figure>
          <img
            src={img}
            class="aspect-video object-cover"
            alt="Guess the borough!"
          />
        </figure>
        <div class="card-body w-100">
          <h3 class="text-center font-bold text-2xl">Guess the borough!</h3>
          <div class="flex flex-col items-center w-100 space-y-2">
            {#each boroughs as borough, i}
              <button
                class="w-4/5 btn {colors[i]} {loading ? 'btn-disabled' : ''}"
                on:click={() => choose(borough)}
              >
                {borough}
              </button>
            {/each}
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>
