export function animationsEnabled(): boolean {
  if (
    process.env["NO_COLOR"] ||
    process.env["SOCA_NO_ANIM"] ||
    process.env["CI"]
  ) {
    return false;
  }
  return Boolean(process.stdout.isTTY);
}
