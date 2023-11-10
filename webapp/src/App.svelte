<script>
  // Packages
  import { v4 as uuidv4 } from "uuid";
  // Data
  import dataset from "./assets/data.json";
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
  const fileServerUrl = "http://localhost:8764/";
  const apiUrl = "http://localhost:4000/submit";
  // State variables
  let page = "welcome";
  let img = null;
  let label = null;
  let img_id = null;
  let loading = false;
  let score = 0;
  let count = 0;
  // User id (store in browser local storage)
  let user_id = localStorage.getItem("user_id");
  if (!user_id) {
    user_id = uuidv4();
    localStorage.setItem("user_id", user_id);
  }

  // Utils
  const formatChoice = (choice) => choice.toLowerCase().replaceAll(/\s/g, "_");
  const randInt = (N) => Math.floor(Math.random() * N);
  const pathToUrl = (filePath) => {
    if (!filePath) return;
    const parts = filePath.split("/");
    const address = parts.slice(2).join("/");
    const url = `${fileServerUrl}${address}`;
    return url;
  };

  // Actions
  const startGame = () => {
    getImg();
    page = "game";
  };

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

    // Save the current label & id
    label = datapoint.borough;
    img_id = datapoint.id;
    if (!label || !img_id) {
      console.error("Loaded datapoint has no label or no id!");
      console.log(datapoint);
      return;
    }

    // Try to serve image from the url
    let url = pathToUrl(datapoint.path);
    if (!url) {
      // Try to serve datapoint url
      console.warn("Url could not be constructed from path.");
      url = datapoint?.url;
      if (!url) {
        console.error("No url could be found for the current datapoint.");
        return;
      }
    }

    // Set image
    img = url;
  };

  const postChoice = async (choice) => {
    // Format
    const fchoice = formatChoice(choice);
    const flabel = formatChoice(label);

    // Increase the count (& score if correct)
    count += 1;
    if (fchoice === flabel) {
      score += 1;
    }

    // Post score to API (data collection)
    const response = await fetch(apiUrl, {
      method: "POST",
      mode: "cors",
      cache: "no-cache",
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
      redirect: "follow",
      referrerPolicy: "no-referrer",
      body: JSON.stringify({
        user_id: user_id,
        img_id: img_id,
        label: flabel,
        guess: fchoice,
      }),
    });
    const body = await response.json();
    if (!body?.data?.success) {
      console.error("Posting failed");
      console.log(body);
    }
  };

  const choose = async (label) => {
    // Disable buttons
    loading = true;

    // Post choice
    await postChoice(label);

    // Get new image
    getImg();

    // Enable buttons
    loading = false;
  };
</script>

<div class="w-100 h-screen flex flex-col items-center p-8">
  <h1 class="text-6xl pb-2">RecogNYCe</h1>
  <h2 class="text-3xl italic pb-2">
    {#if count === 0}
      How well do you know New York City?
    {:else}
      Score: {score} / {count}
    {/if}
  </h2>
  <div class="flex flex-1 flex-col justify-center items-center">
    {#if page === "welcome"}
      <button class="btn btn-lg self-center text-5xl" on:click={startGame}>
        START
      </button>
    {:else if page === "game"}
      <div class="card w-100 bg-base-200 shadow-xl" style="max-width: 600px;">
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
