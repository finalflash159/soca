import { cn } from "@/lib/utils";

interface BrandMarkProps {
  className?: string;
  iconClassName?: string;
  nameClassName?: string;
}

/** The official, decorative parrot mark always travels with the SoCa name. */
export function BrandMark({ className, iconClassName, nameClassName }: BrandMarkProps) {
  return (
    <span className={cn("inline-flex items-center gap-2", className)}>
      <span className={cn("leading-none", iconClassName)} aria-hidden="true">
        🦜
      </span>
      <span className={nameClassName}>SoCa</span>
    </span>
  );
}
