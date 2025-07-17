export default function Home() {
  return (
    <main style={{fontFamily:"sans-serif", maxWidth:800, margin:"2em auto"}}>
      <h1>Music AI Developer Portal</h1>
      <ul>
        <li><a href="/docs">API Docs</a></li>
        <li><a href="/sdk">SDK Usage</a></li>
        <li><a href="/examples">Sample Apps</a></li>
      </ul>
      <section>
        <h2>Get Started</h2>
        <ol>
          <li>Sign up for an API key</li>
          <li>Read the docs</li>
          <li>Try the playground</li>
        </ol>
      </section>
    </main>
  );
}