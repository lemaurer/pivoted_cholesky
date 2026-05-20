# Randomly Pivoted Cholesky Visualizer

Interactive React demo for the geometric interpretation of randomly pivoted Cholesky / Nyström approximation.

## Run locally

```bash
npm install
npm run dev
```

Then open the local URL printed by Vite, usually `http://localhost:5173/`.

## Build locally

```bash
npm run build
npm run preview
```

## Deploy on GitHub Pages

This app is ready to deploy as a static GitHub Pages site.

1. Create a new GitHub repository.
2. Upload or push this folder as the repository contents.
3. In GitHub, open `Settings -> Pages`.
4. Set `Source` to `GitHub Actions`.
5. Push to the `main` branch.

The included workflow at `.github/workflows/deploy.yml` will install dependencies,
run `npm run build`, and publish the generated `dist` folder.

The Vite `base` option is set to `./`, so the app works from a GitHub Pages
repository path such as `https://username.github.io/repository-name/`.
